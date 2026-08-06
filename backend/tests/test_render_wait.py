"""A render that is still running must not be reported as a dead backend.

Seen live, four times in a row. A WAN video finished its 20 sampling steps in
5:04 and then spent ~21 more minutes in VAE decode — `Prompt executed in
00:26:39`. The client's deadline was a flat 600 s, so it raised
BackendUnavailableError while the render was healthy and progressing; the
caller answers that by restarting ComfyUI and retrying at reduced settings, so
the job walked 704x1280 down to 256x256 and threw away four good renders.

The rule: keep waiting while ComfyUI says this prompt is still queued or
running; give up when it is gone, unreachable, or the absolute cap is hit.
"""
import json
import time
import unittest

from app.adapters.base import BackendUnavailableError
from app.adapters.comfyui import ComfyUIClient, WorkflowRuntimeError


class FakeClient(ComfyUIClient):
    """A client whose HTTP layer is scripted, so no server is needed."""

    def __init__(self, queue_answers, history_answers, **kw):
        super().__init__("http://127.0.0.1:8188", poll_interval=0.0,
                         timeout_s=0.01, **kw)
        self.queue_answers = list(queue_answers)
        self.history_answers = list(history_answers)
        self.queue_calls = 0

    def request(self, method, path, data=None, headers=None):  # noqa: ARG002
        if path == "/queue":
            self.queue_calls += 1
            answer = (self.queue_answers.pop(0) if self.queue_answers
                      else {"queue_running": [], "queue_pending": []})
            if isinstance(answer, Exception):
                raise answer
            return json.dumps(answer).encode()
        if path.startswith("/history/"):
            # Outlast the (deliberately tiny) deadline, so every poll forces
            # the "is it still executing?" decision this suite is about.
            time.sleep(0.005)
            answer = (self.history_answers.pop(0) if self.history_answers
                      else {})
            if isinstance(answer, Exception):
                raise answer
            return json.dumps(answer).encode()
        if path.startswith("/view"):
            return b"PNGDATA"
        raise AssertionError(f"unexpected path {path}")


RUNNING = {"queue_running": [[0, "pid-1", {}, {}, []]], "queue_pending": []}
IDLE = {"queue_running": [], "queue_pending": []}
DONE = {"pid-1": {"status": {"status_str": "success"},
                  "outputs": {"9": {"images": [
                      {"filename": "out.webp", "subfolder": "",
                       "type": "output"}]}}}}


class StillExecuting(unittest.TestCase):

    def test_a_running_prompt_is_recognised(self):
        c = FakeClient([RUNNING], [])
        self.assertTrue(c._still_executing("pid-1"))

    def test_a_pending_prompt_counts_too(self):
        c = FakeClient([{"queue_running": [],
                         "queue_pending": [[1, "pid-1", {}, {}, []]]}], [])
        self.assertTrue(c._still_executing("pid-1"))

    def test_somebody_elses_prompt_does_not_count(self):
        c = FakeClient([{"queue_running": [[0, "other", {}, {}, []]],
                         "queue_pending": []}], [])
        self.assertFalse(c._still_executing("pid-1"))

    def test_an_empty_queue_means_not_working(self):
        self.assertFalse(FakeClient([IDLE], [])._still_executing("pid-1"))

    def test_an_unreachable_comfyui_means_not_working(self):
        """If the queue cannot be read, the backend really may be gone —
        that is the one case where giving up is correct."""
        c = FakeClient([BackendUnavailableError("refused")], [])
        self.assertFalse(c._still_executing("pid-1"))

    def test_a_malformed_queue_entry_does_not_raise(self):
        c = FakeClient([{"queue_running": ["nonsense", None, 7],
                         "queue_pending": []}], [])
        self.assertFalse(c._still_executing("pid-1"))


class WaitingForALongRender(unittest.TestCase):

    def test_a_slow_but_running_render_is_waited_out(self):
        """The deadline expires twice before the output appears; because the
        prompt is still executing, the client keeps waiting instead of
        declaring the backend dead."""
        c = FakeClient([RUNNING, RUNNING], [{}, {}, DONE])
        c._MAX_WAIT_MULTIPLE = 100_000  # the cap has its own test
        data, name = c.wait_for_output_file("pid-1")
        self.assertEqual(data, b"PNGDATA")
        self.assertEqual(name, "out.webp")
        self.assertGreaterEqual(c.queue_calls, 1)

    def test_a_vanished_prompt_still_gives_up(self):
        c = FakeClient([IDLE], [{}])
        c._MAX_WAIT_MULTIPLE = 100_000  # give-up must come from the queue
        with self.assertRaises(BackendUnavailableError):
            c.wait_for_output_file("pid-1")

    def test_it_cannot_wait_for_ever(self):
        """A queue that reports the prompt forever must still hit a cap, or a
        wedged ComfyUI would hang the job permanently."""
        c = FakeClient([RUNNING] * 500, [{}] * 500)
        c._MAX_WAIT_MULTIPLE = 0  # cap already exceeded on the first check
        with self.assertRaises(BackendUnavailableError):
            c.wait_for_output_file("pid-1")

    def test_an_execution_error_is_still_reported_as_one(self):
        """A real ComfyUI error must stay a WorkflowRuntimeError — the new
        patience must not swallow genuine failures."""
        c = FakeClient([RUNNING], [{"pid-1": {
            "status": {"status_str": "error", "messages": ["boom"]}}}])
        with self.assertRaises(WorkflowRuntimeError):
            c.wait_for_output_file("pid-1")




class BriefSilenceIsNotDeath(unittest.TestCase):
    """ComfyUI stops answering HTTP for a stretch under memory pressure and
    then comes back. The caller answers "gone" by RESTARTING it and throwing
    the render away, so one refused connection must not count as death — only
    sustained silence may. This is the path that ended a healthy render at
    7m54s even after the deadline was fixed."""

    def make(self, history_answers):
        c = FakeClient([RUNNING] * 50, history_answers)
        c._MAX_WAIT_MULTIPLE = 100_000
        return c

    def test_a_blip_is_ridden_out(self):
        refused = BackendUnavailableError("connection refused")
        c = self.make([refused, refused, DONE])
        data, _name = c.wait_for_output_file("pid-1")
        self.assertEqual(data, b"PNGDATA")

    def test_sustained_silence_is_still_reported(self):
        c = self.make([BackendUnavailableError("refused")] * 50)
        c._UNREACHABLE_GRACE_S = 0.0  # every refusal is already "sustained"
        with self.assertRaises(BackendUnavailableError):
            c.wait_for_output_file("pid-1")

    def test_recovery_resets_the_silence_clock(self):
        """A blip, a good answer, then another blip must not add up to a
        sustained outage."""
        refused = BackendUnavailableError("refused")
        c = self.make([refused, {}, refused, DONE])
        data, _name = c.wait_for_output_file("pid-1")
        self.assertEqual(data, b"PNGDATA")


if __name__ == "__main__":
    unittest.main()
