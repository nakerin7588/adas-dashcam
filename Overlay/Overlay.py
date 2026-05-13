import cv2
import numpy as np

class Overlay:
    def __init__(self):
        # UI Configuration
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        self.alpha = 0.6  # Panel transparency (0.0 to 1.0)
        
        # Color Palette (BGR format for OpenCV)
        self.colors = {
            "safe": (100, 220, 100),     # Green
            "warning": (0, 165, 255),    # Orange
            "danger": (50, 50, 255),     # Red
            "text": (255, 255, 255),     # White text
            "panel_bg": (30, 30, 30)     # Dark gray panel
        }

    def _draw_transparent_panel(self, frame, text, text_color, bg_color, top_left_x, top_left_y):
        """Draws a semi-transparent box with text using optimized ROI blending."""
        # Calculate text size to dynamically size the background panel
        text_size, _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
        text_w, text_h = text_size
        
        # Panel padding
        pad_x, pad_y = 15, 10
        
        # Define panel coordinates
        x1, y1 = top_left_x, top_left_y
        x2, y2 = x1 + text_w + (pad_x * 2), y1 + text_h + (pad_y * 2)
        
        # Ensure we don't draw outside the frame boundaries
        h, w = frame.shape[:2]
        # if x2 > w or y2 > h:
        #     return frame

        # Extract the Region of Interest (ROI)
        roi = frame[y1:y2, x1:x2]
        
        # Create a colored panel of the same size
        panel = np.full_like(roi, bg_color, dtype=np.uint8)
        
        # Blend the panel and the ROI
        blended = cv2.addWeighted(panel, self.alpha, roi, 1 - self.alpha, 0)
        frame[y1:y2, x1:x2] = blended
        
        # Put the text on top (fully opaque)
        text_org = (x1 + pad_x, y2 - pad_y)
        cv2.putText(frame, text, text_org, self.font, self.font_scale, text_color, self.thickness, cv2.LINE_AA)
        
        # return frame

    def Draw(self, frame, ldws_state, fcws_state) -> None:
        """
        Renders both overlays onto the frame.
        States: 0 = Safe, 1 = Warning, 2 = Danger
        """
        h, w = frame.shape[:2]
        
        # --- LDWS Logic (Top Left) ---
        ldws_text = "LDWS: TRACKING"
        ldws_color = self.colors["safe"]
        
        if ldws_state == 1:
            ldws_text = "LDWS: SHIFTING!"
            ldws_color = self.colors["warning"]
        elif ldws_state == 2:
            ldws_text = "LDWS: DEPARTURE!"
            ldws_color = self.colors["danger"]
            
        # Draw LDWS at top-left (20px margin)
        self._draw_transparent_panel(frame, ldws_text, self.colors["text"], ldws_color, 20, 20)
        
        # --- FCWS Logic (Top Right) ---
        fcws_text = "FCWS: CLEAR"
        fcws_color = self.colors["safe"]
        
        if fcws_state == 1:
            fcws_text = "FCWS: CAR AHEAD"
            fcws_color = self.colors["warning"]
        # elif fcws_state == 2:
        #     fcws_text = "FCWS: BRAKE!"
        #     fcws_color = self.colors["danger"]
            
        # Calculate X position for top-right to keep it right-aligned
        text_size, _ = cv2.getTextSize(fcws_text, self.font, self.font_scale, self.thickness)
        fcws_x = w - text_size[0] - 50 # 50px margin from the right edge
        
        # Draw FCWS at top-right
        self._draw_transparent_panel(frame, fcws_text, self.colors["text"], fcws_color, fcws_x, 20)
        
        # return frame