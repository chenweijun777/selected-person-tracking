"""Thread-safe GUI bridge; all model state remains on the inference thread."""

from __future__ import annotations

from queue import Empty, Queue


class GuiControl:
    def __init__(self):
        self.commands = Queue()
        self.frames = Queue(maxsize=1)
        self.events = Queue()

    def send(self, action, **values):
        self.commands.put({"action": action, **values})

    def poll(self, timeout=0.0):
        try:
            commands = [self.commands.get(timeout=timeout)]
        except Empty:
            return []
        while True:
            try:
                commands.append(self.commands.get_nowait())
            except Empty:
                return commands

    def present(self, canvas, **values):
        item = {"canvas": canvas, **values}
        try:
            self.frames.get_nowait()
        except Empty:
            pass
        self.frames.put_nowait(item)

    def feedback(self, event):
        self.events.put(event)


def image_viewport(image_size, canvas_size):
    """Aspect-preserving fit and offset, shared by rendering and click mapping."""
    width, height = image_size
    canvas_width, canvas_height = canvas_size
    if min(width, height, canvas_width, canvas_height) <= 0:
        raise ValueError("Image and canvas dimensions must be positive")
    scale = min(canvas_width / width, canvas_height / height)
    shown_width = max(1, round(width * scale))
    shown_height = max(1, round(height * scale))
    return (
        (canvas_width - shown_width) // 2,
        (canvas_height - shown_height) // 2,
        shown_width,
        shown_height,
    )


def image_point(point, image_size, viewport):
    """Letterbox clicks return None; use actual rounded display dimensions."""
    x, y = point
    left, top, width, height = viewport
    if not (left <= x < left + width and top <= y < top + height):
        return None
    image_width, image_height = image_size
    return (x - left) * image_width / width, (y - top) * image_height / height
