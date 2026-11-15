"""API tests: the endpoints that do not need weights loaded.

The checkpoints are gitignored (68 MB each), so CI can only exercise the paths
that run before inference: class tables, the correction validation and the
upload size guard.  Everything that needs weights is covered by the manual
end-to-end runs documented in results/REPORT_*.md.
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from foodcls import app as app_module  # noqa: E402
from foodcls.classes import CLASS_NAMES  # noqa: E402

# No context manager: entering it would run the lifespan and load three
# checkpoints, which CI does not have.
client = TestClient(app_module.app)


def jpeg_bytes(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (140, 90, 60)).save(buf, format="JPEG")
    return buf.getvalue()


class ClassTableApiTest(unittest.TestCase):
    def test_classes_endpoint_matches_the_table(self):
        r = client.get("/api/classes")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["count"], len(CLASS_NAMES))
        self.assertEqual([c["class"] for c in body["classes"]], CLASS_NAMES)

    def test_wide_classes_endpoint(self):
        r = client.get("/api/wide/classes")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["count"], len(body["classes"]))
        self.assertGreater(body["count"], len(CLASS_NAMES))


class FeedbackApiTest(unittest.TestCase):
    def test_unknown_class_is_rejected(self):
        r = client.post("/api/feedback", data={"sample": "x.jpg", "corrected": "sushi_roll"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("未知的类别", r.json()["error"])

    def test_nothing_marked_is_rejected(self):
        r = client.post("/api/feedback", data={"sample": "x.jpg", "corrected": "", "removed": ""})
        self.assertEqual(r.status_code, 400)
        self.assertIn("请先选出正确的菜", r.json()["error"])

    def test_removal_alone_is_valid(self):
        # Validation comes before the image is resolved, so a strike-out-only
        # submission with a bogus sample name must fail on the lookup (404),
        # not on "nothing was marked" (400).
        r = client.post("/api/feedback", data={"sample": "x.jpg", "corrected": "", "removed": "pizza"})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["error"], "not found")


class UploadGuardTest(unittest.TestCase):
    def test_oversized_image_is_refused(self):
        # A 40 MP JPEG of flat colour is a few hundred KB, so the byte limit
        # never sees it; the pixel guard is what stops the decode.
        old = app_module.MAX_PIXELS
        app_module.MAX_PIXELS = 1000
        try:
            r = client.post("/api/feedback",
                            data={"sample": "x.jpg", "corrected": "pizza"},
                            files={"file": ("big.jpg", jpeg_bytes((64, 64)), "image/jpeg")})
        finally:
            app_module.MAX_PIXELS = old
        self.assertEqual(r.status_code, 413)


if __name__ == "__main__":
    unittest.main()
