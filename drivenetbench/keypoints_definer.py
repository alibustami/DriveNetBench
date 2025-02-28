"""Utility functions for Easier POC applications."""

import tkinter
from tkinter import Canvas, Tk
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageTk

import drivenetbench.utilities.config as configs
from drivenetbench.utilities.utils import path_checker


class KeyPointsDefiner:
    """Class for defining geometry on a video or image source."""

    def __init__(
        self,
        source: Optional[str] = None,
        output_name: Optional[str] = None,
        frame_num: Optional[int] = None,
        override_if_exists: Optional[bool] = False,
    ):
        """Initialize the KeyPointsDefiner.

        Parameters
        ----------
        source : str
            The source of the video or image.
        output_name : str
            The name of the output numpy file.
        frame_num : int, optional
            The frame number to extract for videos (default is 1).
        override_if_exists : bool, optional
            Whether to override the output file if it exists (default is False).
        """
        self.source = (
            source
            if source is not None
            else configs.get_config("keypoints_definer.source_path")
        )
        self.output_name = (
            output_name
            if output_name is not None
            else configs.get_config("keypoints_definer.output_name")
        )
        self.frame_num = (
            frame_num
            if frame_num is not None
            else configs.get_config("keypoints_definer.frame_number")
        )
        self.override_if_exists = (
            override_if_exists
            if override_if_exists
            else configs.get_config("keypoints_definer.override_if_exists")
        )

        if path_checker(f"{self.output_name}.npy", break_if_not_found=False)[
            0
        ]:
            if not override_if_exists:
                raise FileExistsError(
                    f"Output file {self.output_name}.npy already exists. "
                    "Use the --override-if-exists flag to override."
                )

        self.frame_num = frame_num

        self.window = None
        self.canvas = None
        self.points = []
        self.resize_factor = 1
        self.original_frame_width = None
        self.original_frame_height = None
        self.tk_image = None

    def run(self):
        """Run the geometry definer application."""
        self.setup_window()
        frame = self.extract_frame()
        self.setup_canvas(frame)
        self.setup_bindings()
        self.window.mainloop()

    def setup_window(self):
        """Set up the Tkinter window."""
        self.window = Tk()
        self.window.title("Define Geometry")

    def extract_frame(self) -> npt.NDArray:
        """Extract a frame from the source.

        Returns
        -------
        npt.NDArray
            The extracted frame in RGB format.

        Raises
        ------
        Exception
            If the frame cannot be read.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise Exception(f"Failed to open source: {self.source}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 1:
            self.frame_num = 0
        else:
            self.frame_num = min(max(0, self.frame_num), total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise Exception(f"Failed to read frame {self.frame_num}.")

        self.original_frame_height, self.original_frame_width = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return frame

    def setup_canvas(self, frame: npt.NDArray):
        """Set up the Tkinter canvas with the frame.

        Parameters
        ----------
        frame : npt.NDArray
            The frame to display.
        """
        screen_pad = 200
        screen_width = self.window.winfo_screenwidth() - screen_pad
        screen_height = self.window.winfo_screenheight() - screen_pad

        self.resize_factor = min(
            screen_width / self.original_frame_width,
            screen_height / self.original_frame_height,
        )

        canvas_pad = 100
        window_width = (
            int(self.original_frame_width * self.resize_factor) + canvas_pad
        )
        window_height = (
            int(self.original_frame_height * self.resize_factor) + canvas_pad
        )

        frame = cv2.resize(
            frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor
        )
        self.window.geometry(f"{window_width}x{window_height}")

        resized_height, resized_width = frame.shape[:2]
        image = Image.fromarray(frame)
        self.tk_image = ImageTk.PhotoImage(image)

        self.canvas = Canvas(
            self.window, width=resized_width, height=resized_height
        )
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.pack()

    def setup_bindings(self):
        """Set up event bindings for the canvas."""
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Key>", self.on_key_press)
        self.canvas.focus_set()

    def on_click(self, event: tkinter.Event):
        """Handle mouse click events.

        Parameters
        ----------
        event : tkinter.Event
            The event object.
        """
        x, y = event.x, event.y
        self.points.append((x, y))
        self.update_canvas()

    def on_key_press(self, event: tkinter.Event):
        """Handle key press events.

        Parameters
        ----------
        event : tkinter.Event
            The event object.
        """
        if event.char.lower() == "q":
            if self.points:
                self.save_points()
            self.window.destroy()
        elif event.char.lower() == "z":
            if self.points:
                self.points.pop()
                self.update_canvas()

    def save_points(self):
        """Save the collected points to a numpy file."""
        points_arr = np.array(self.points)
        scale_factor = 1 / self.resize_factor
        points_arr = points_arr * scale_factor
        points_arr = points_arr.astype(int)

        points_arr[:, 0] = np.clip(
            points_arr[:, 0], 0, self.original_frame_width - 1
        )
        points_arr[:, 1] = np.clip(
            points_arr[:, 1], 0, self.original_frame_height - 1
        )

        np.save(f"{self.output_name}.npy", points_arr)

    def update_canvas(self):
        """Update the canvas by redrawing the points and shapes."""
        self.canvas.delete("points")

        for i, point in enumerate(self.points):
            x, y = point
            radius = 3
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="red",
                outline="red",
                tag="points",
            )
            # write the point index
            self.canvas.create_text(
                x + 10,
                y + 10,
                text=str(i + 1),
                fill="red",
                tag="points",
            )
