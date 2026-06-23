from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv, Pose6DOF


def _encode_image(rgb: np.ndarray) -> str:
    """Encode a numpy RGB array to a base64 JPEG string for API calls."""
    img = Image.fromarray(rgb.astype(np.uint8))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _scene_hash(rgb: np.ndarray) -> str:
    return hashlib.md5(rgb.tobytes()).hexdigest()


class PerceptionModule:
    """Two-stage perception pipeline.

    Stage 1 — VLM semantic grounding: calls GPT-4o Vision (or local LLaVA)
    to identify objects and their 2-D pixel positions in the scene.

    Stage 2 — Geometric pose estimation: back-projects the 2-D pixel +
    depth value to a 3-D world-frame pose using the camera intrinsics and
    extrinsics supplied by CobotEnv.
    """

    # Colour names the planner understands
    _COLOURS = ("red", "green", "blue", "yellow", "orange", "purple")

    # Tokens that signal an ambiguous reference — VLM resolution needed
    _AMBIGUOUS_RE = re.compile(
        r"\b(it|that|this|those|these|"
        r"the\s+one|the\s+block|the\s+cube|the\s+thing|"
        r"closest|nearest|furthest|farthest|"
        r"left\s+one|right\s+one|the\s+other|"
        r"first|second|third|last)\b",
        re.IGNORECASE,
    )

    _RESOLVE_PROMPT = (
        "You are a robot arm assistant. You can see the tabletop through the robot's camera.\n"
        "The user has given a voice command that contains ambiguous references — pronouns, "
        "spatial descriptions, or object references without explicit colour names.\n\n"
        "Your job: look at the image, identify which coloured block the user is referring to, "
        "and rewrite the command replacing every ambiguous reference with the specific colour name.\n\n"
        "Rules:\n"
        "- Only use these colour names: red, green, blue, yellow, orange, purple.\n"
        "- If a colour is already explicit, keep it unchanged.\n"
        "- If you cannot tell which object is meant, keep that part of the command unchanged.\n"
        "- Return ONLY the rewritten command as a plain sentence — no explanation, no punctuation changes.\n\n"
        "Example:\n"
        "  Visible objects: red cube, green cube\n"
        "  Command: 'pick up that one and put it on the other'\n"
        "  Response: 'pick up the red cube and put it on the green cube'\n"
    )

    _SYSTEM_PROMPT = (
        "You are a robot perception system. Given a top-down RGB image of a tabletop scene, "
        "identify all visible coloured blocks and cylinders on the table surface. "
        "Ignore the robot arm, gripper, and any mechanical parts — focus only on the coloured objects.\n"
        "Return ONLY a raw JSON object (no prose, no markdown, no explanation) with this exact structure:\n"
        '{"objects": [{"id": "<color>_<shape>", "color": "<color>", '
        '"shape": "cube|cylinder", "pixel_u": <int>, "pixel_v": <int>}]}\n'
        "pixel_u is the horizontal pixel coordinate (left=0), "
        "pixel_v is the vertical pixel coordinate (top=0). "
        "Use colour names: red, blue, green, yellow, orange, purple. "
        "If no coloured objects are visible, return: {\"objects\": []}. "
        "Output ONLY the JSON — no other text."
    )

    def __init__(self, config: dict, env: "CobotEnv") -> None:
        self._config = config
        self._env = env
        self._provider = config.get("vlm_provider", "openai")
        self._cache_threshold = config.get("cache_threshold", 0.05)
        self._cache: dict[str, dict] = {}  # scene_hash → scene description

        if self._provider == "groq":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
            )
            self._model = config.get("vlm_model", "meta-llama/llama-4-scout-17b-16e-instruct")
        elif self._provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self._model = config.get("vlm_model", "gpt-4o")
        else:
            self._client = None
            self._model = config.get("local_vlm_model", "llava")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Command reference resolution
    # ------------------------------------------------------------------

    def resolve_command(
        self,
        command: str,
        rgb: np.ndarray,
        available_colours: list[str],
    ) -> str:
        """Rewrite *command* so every ambiguous object reference becomes a
        concrete colour name.

        Skips the VLM call entirely when the command already names at least
        one colour AND contains no ambiguous pronouns/spatial tokens — keeps
        latency at zero for unambiguous commands like "push the red block left".

        Args:
            command:           raw user command string
            rgb:               current agentview camera frame (HxWx3 uint8)
            available_colours: colours currently on the table (e.g. ["red","green","blue"])

        Returns:
            Rewritten command string (falls back to original on any error).
        """
        if not self._needs_resolution(command):
            return command

        if self._client is None:
            log.debug("[resolve] no VLM client — returning command unchanged")
            return command

        colour_list = ", ".join(available_colours) if available_colours else "red, green"
        user_text = (
            f"Visible objects on the table: {colour_list} cube(s).\n"
            f"Command: \"{command}\""
        )

        try:
            b64 = _encode_image(rgb)
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._RESOLVE_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
                max_tokens=128,
                temperature=0.0,
            )
            resolved = response.choices[0].message.content.strip().strip('"')
            if resolved:
                log.info("[resolve] '%s' → '%s'", command, resolved)
                return resolved
        except Exception as exc:
            log.warning("[resolve] VLM call failed (%s) — using original command", exc)

        return command

    def _needs_resolution(self, command: str) -> bool:
        """Return True when the command has no colour word OR has ambiguous tokens."""
        cmd = command.lower()
        has_colour = any(c in cmd for c in self._COLOURS)
        has_ambiguous = bool(self._AMBIGUOUS_RE.search(cmd))
        return has_ambiguous or not has_colour

    def get_scene_description(self, rgb: np.ndarray, max_retries: int = 3) -> dict:
        """Return a JSON scene description from the VLM.

        Results are cached by scene image hash. Retries up to max_retries times
        if the VLM returns an empty object list (common when it responds in prose).
        """
        h = _scene_hash(rgb)
        if h in self._cache:
            return self._cache[h]

        for attempt in range(max_retries):
            if self._provider in ("openai", "groq"):
                scene = self._query_openai(rgb)
            else:
                scene = self._query_local(rgb)

            if scene.get("objects"):
                break
            if attempt < max_retries - 1:
                log.warning("[perception] empty scene on attempt %d, retrying...", attempt + 1)

        self._cache[h] = scene
        return scene

    def get_object_pose(
        self,
        obj_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_name: str = "agentview",
    ) -> "Pose6DOF":
        """Return the 6-DOF world-frame pose of an object.

        Uses the VLM-detected 2-D pixel position and the depth image to
        back-project into 3-D, then applies camera extrinsics.
        """
        from cobot.env.cobot_env import Pose6DOF

        scene = self.get_scene_description(rgb)
        obj = self._find_object(scene, obj_id)

        u, v = int(obj["pixel_u"]), int(obj["pixel_v"])
        u = np.clip(u, 0, depth.shape[1] - 1)
        v = np.clip(v, 0, depth.shape[0] - 1)

        d = float(depth[v, u])
        if d <= 0:
            # Fallback: sample a small patch around the pixel
            patch = depth[max(0, v - 5):v + 5, max(0, u - 5):u + 5]
            valid = patch[patch > 0]
            d = float(valid.mean()) if valid.size > 0 else 0.5

        K = self._env.get_camera_intrinsics(camera_name)
        E = self._env.get_camera_extrinsics(camera_name)  # 4×4 world-to-cam

        p_world = self._backproject(u, v, d, K, E)
        # Orientation from perception is not reliable at this stage; use identity
        return Pose6DOF(p_world[0], p_world[1], p_world[2], 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # VLM backends
    # ------------------------------------------------------------------

    def _query_openai(self, rgb: np.ndarray) -> dict:
        b64 = _encode_image(rgb)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": "Describe the scene."},
                    ],
                },
            ],
            max_tokens=512,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        log.debug("[perception] VLM raw: %s", raw[:300])
        return self._parse_scene_json(raw)

    def _query_local(self, rgb: np.ndarray) -> dict:
        """Query a locally hosted LLaVA model via Ollama REST API."""
        import requests

        b64 = _encode_image(rgb)
        payload = {
            "model": self._model,
            "prompt": self._SYSTEM_PROMPT + "\n\nDescribe the scene.",
            "images": [b64],
            "stream": False,
        }
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        return self._parse_scene_json(raw)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _backproject(
        u: int, v: int, depth: float, K: np.ndarray, E: np.ndarray
    ) -> np.ndarray:
        """Back-project a pixel + depth to 3-D world coordinates.

        Args:
            u, v: pixel coordinates (u = column, v = row)
            depth: depth in metres at (u, v)
            K: 3×3 intrinsic matrix
            E: 4×4 world-to-camera extrinsic matrix
        Returns:
            3-D point in world frame
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Point in camera frame (camera looking along +Z)
        x_c = (u - cx) * depth / fx
        y_c = (v - cy) * depth / fy
        p_cam = np.array([x_c, y_c, depth, 1.0])

        # E maps world → camera; E^{-1} maps camera → world
        E_inv = np.linalg.inv(E)
        p_world = E_inv @ p_cam
        return p_world[:3]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_scene_json(raw: str) -> dict:
        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # VLM sometimes wraps JSON in prose — extract the outermost {...} block
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        log.warning("[perception] JSON parse failed — raw: %r", raw[:500])
        return {"objects": []}

    @staticmethod
    def _find_object(scene: dict, obj_id: str) -> dict:
        for obj in scene.get("objects", []):
            if obj.get("id") == obj_id:
                return obj
            # Fuzzy match: "red_block" matches {"color": "red", "shape": "cube"}
            color, _, shape = obj_id.partition("_")
            shape_map = {"block": "cube", "cube": "cube", "cylinder": "cylinder"}
            if (
                obj.get("color") == color
                and shape_map.get(shape) == obj.get("shape")
            ):
                return obj
        raise ValueError(f"Object '{obj_id}' not found in scene: {scene}")
