import moderngl
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

class UIRenderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._load_shaders()
        self._create_quad()
        
        self.pointers = []
        
        # Define 3 buttons: Background, Theme, WetScreen
        # coordinates are NDC: [-1, 1], max X is around 0.75 due to aspect ratio scaling in tracker
        self.buttons = [
            {
                "id": "bg",
                "label": "Camera",
                "color": (0.8, 0.2, 0.2, 0.6), # Red
                "active_color": (1.0, 0.3, 0.3, 0.9),
                "pos": (-0.4, 0.75),
                "size": (0.15, 0.08),
                "hover_time": 0.0,
                "triggered": False
            },
            {
                "id": "theme",
                "label": "Theme",
                "color": (0.2, 0.8, 0.2, 0.6), # Green
                "active_color": (0.3, 1.0, 0.3, 0.9),
                "pos": (0.0, 0.75),
                "size": (0.15, 0.08),
                "hover_time": 0.0,
                "triggered": False
            },
            {
                "id": "wet",
                "label": "Draw",
                "color": (0.2, 0.2, 0.8, 0.6), # Blue
                "active_color": (0.3, 0.3, 1.0, 0.9),
                "pos": (0.4, 0.75),
                "size": (0.15, 0.08),
                "hover_time": 0.0,
                "triggered": False
            }
        ]
        
        self.trigger_threshold = 15 # frames to trigger (approx 0.5 sec at 30fps)
        self._create_textures()

    def _load_shaders(self):
        base_path = Path('assets/shaders')
        with open(base_path / 'ui.vert', 'r') as f:
            vert = f.read()
        with open(base_path / 'ui.frag', 'r') as f:
            frag = f.read()
        self.prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)

    def _create_quad(self):
        # A quad centered at 0,0 with size 1x1 
        # So we can scale it by size and translate by pos
        # We also need texcoords for the text texture
        vertices = np.array([
            # x, y, u, v
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 0.0,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])

    def _create_textures(self):
        # Generate text textures for each button using PIL
        for btn in self.buttons:
            # Create a transparent image
            img = Image.new('RGBA', (200, 100), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            # Try to get a font, fallback to default
            try:
                # Use a larger font
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()
                
            text = btn["label"]
            # Center text
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = (200 - w) / 2
            y = (100 - h) / 2 - bbox[1]
            draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
            
            # Create moderngl texture
            tex = self.ctx.texture(img.size, 4, img.tobytes())
            btn["texture"] = tex

        # Generate cursor texture (hollow circle)
        cursor_img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        cursor_draw = ImageDraw.Draw(cursor_img)
        # Draw a thick hollow circle
        cursor_draw.ellipse([4, 4, 60, 60], outline=(255, 255, 255, 255), width=6)
        # Add an inner dot
        cursor_draw.ellipse([28, 28, 36, 36], fill=(255, 255, 255, 255))
        self.cursor_texture = self.ctx.texture(cursor_img.size, 4, cursor_img.tobytes())

    def update(self, hands_data) -> list[str]:
        """
        Checks if any index finger tip is hovering over a button.
        Returns a list of button IDs that were triggered this frame.
        """
        triggered_events = []
        
        # Get all index finger positions (NDC)
        self.pointers = []
        if hands_data:
            for hand in hands_data:
                self.pointers.append(hand[8]) # Index finger tip
                
        for btn in self.buttons:
            is_hovered = False
            for p in self.pointers:
                # Check AABB with slightly larger bounds for easier clicking
                bx, by = btn["pos"]
                bw, bh = btn["size"]
                
                # expand bounds slightly for forgiving interaction
                if (bx - bw*1.5) <= p[0] <= (bx + bw*1.5) and (by - bh*2.0) <= p[1] <= (by + bh*2.0):
                    is_hovered = True
                    break
                    
            if is_hovered:
                btn["hover_time"] += 1
                if btn["hover_time"] == 1:
                    from loguru import logger
                    logger.debug(f"Hover started on {btn['id']}")
                    
                if btn["hover_time"] >= self.trigger_threshold:
                    if not btn["triggered"]:
                        triggered_events.append(btn["id"])
                        btn["triggered"] = True
            else:
                # Decay hover time slowly so a dropped frame doesn't instantly reset it
                btn["hover_time"] = max(0, btn["hover_time"] - 1)
                if btn["hover_time"] == 0:
                    btn["triggered"] = False
                
        return triggered_events

    def render(self):
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.DEFAULT_BLENDING
        
        # Render buttons
        if 'use_texture' in self.prog:
            self.prog['use_texture'].value = True
            
        for btn in self.buttons:
            self.prog['rect_pos'].value = btn["pos"]
            self.prog['rect_size'].value = btn["size"]
            
            # Pulse logic or hover color based on progress
            progress = min(1.0, btn["hover_time"] / self.trigger_threshold)
            
            # Interpolate between normal and active color based on hover time
            c1 = np.array(btn["color"])
            c2 = np.array(btn["active_color"])
            current_color = c1 + (c2 - c1) * progress
            self.prog['rect_color'].value = tuple(current_color)
            
            # Bind the text texture
            btn["texture"].use(0)
            if 'text_texture' in self.prog:
                self.prog['text_texture'].value = 0
                
            self.vao.render(moderngl.TRIANGLE_STRIP)

        # Render pointers (cursors)
        if 'use_texture' in self.prog:
            self.prog['use_texture'].value = True
            
        self.prog['rect_color'].value = (1.0, 1.0, 1.0, 0.9)
        self.cursor_texture.use(0)
        if 'text_texture' in self.prog:
            self.prog['text_texture'].value = 0
            
        for p in self.pointers:
            self.prog['rect_pos'].value = tuple(p)
            self.prog['rect_size'].value = (0.02, 0.02 * (1280/720)) # approximate square aspect ratio compensation
            self.vao.render(moderngl.TRIANGLE_STRIP)
