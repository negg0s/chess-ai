"""
Chessboard Computer Vision - Board Detector
Detects and crops an 8x8 chessboard from screenshots or images,
and slices it into 64 individual square patches.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional

class BoardDetector:
    def __init__(self):
        pass

    def detect_board_contour(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Attempts to find the square chessboard contour using edge & contour detection.
        Returns 4 corner points of the board.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(cnt) > (img.shape[0] * img.shape[1] * 0.1):
                return approx

        return None

    def warp_board(self, img: np.ndarray, corners: np.ndarray, output_size: int = 800) -> np.ndarray:
        """
        Perspective warp of the chessboard to a standard square (output_size x output_size).
        """
        pts = corners.reshape(4, 2).astype(np.float32)
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        dst = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, matrix, (output_size, output_size))
        return warped

    def slice_into_squares(self, board_img: np.ndarray) -> List[List[np.ndarray]]:
        """
        Slices an 800x800 chessboard into an 8x8 grid (64 square images).
        Returns matrix of shape [8][8], where row 0 is Rank 8 (top), col 0 is File a (left).
        """
        h, w = board_img.shape[:2]
        sq_h = h // 8
        sq_w = w // 8
        grid = []

        for r in range(8):
            row_squares = []
            for c in range(8):
                sq = board_img[r * sq_h:(r + 1) * sq_h, c * sq_w:(c + 1) * sq_w]
                row_squares.append(sq)
            grid.append(row_squares)

        return grid
