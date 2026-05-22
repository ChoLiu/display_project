"""Display HAT Mini smoke test: draws status to LCD, reads buttons, drives RGB LED.

A=red  B=green  X=blue  Y=white  (no button → LED off)
Ctrl-C to quit.
"""
import time
from PIL import Image, ImageDraw, ImageFont
from displayhatmini import DisplayHATMini

WIDTH, HEIGHT = DisplayHATMini.WIDTH, DisplayHATMini.HEIGHT

buffer = Image.new("RGB", (WIDTH, HEIGHT))
draw = ImageDraw.Draw(buffer)
dhm = DisplayHATMini(buffer)

try:
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except OSError:
    font_big = font_sm = ImageFont.load_default()

BUTTONS = [
    ("A", DisplayHATMini.BUTTON_A, (255, 0, 0)),
    ("B", DisplayHATMini.BUTTON_B, (0, 255, 0)),
    ("X", DisplayHATMini.BUTTON_X, (0, 0, 255)),
    ("Y", DisplayHATMini.BUTTON_Y, (255, 255, 255)),
]

print("Running. Press buttons on the HAT, Ctrl-C to quit.")
try:
    while True:
        pressed = [(name, color) for name, pin, color in BUTTONS if dhm.read_button(pin)]

        draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(20, 20, 30))
        draw.text((10, 8), "Display HAT Mini", font=font_big, fill=(255, 255, 255))
        draw.text((10, 48), f"{time.strftime('%H:%M:%S')}", font=font_sm, fill=(180, 180, 180))

        # Four button indicators
        for i, (name, pin, color) in enumerate(BUTTONS):
            x = 10 + i * 78
            y = 90
            on = dhm.read_button(pin)
            box = (x, y, x + 70, y + 70)
            draw.rectangle(box, fill=color if on else (50, 50, 60), outline=(200, 200, 200))
            tcolor = (0, 0, 0) if on else (200, 200, 200)
            draw.text((x + 26, y + 22), name, font=font_big, fill=tcolor)

        status = "Press a button" if not pressed else "+ ".join(n for n, _ in pressed)
        draw.text((10, 180), status, font=font_sm, fill=(255, 220, 100))
        draw.text((10, 210), "Ctrl-C to quit", font=font_sm, fill=(120, 120, 120))

        if pressed:
            r, g, b = pressed[-1][1]
            dhm.set_led(r / 255, g / 255, b / 255)
        else:
            dhm.set_led(0, 0, 0)

        dhm.display()
        time.sleep(0.03)
except KeyboardInterrupt:
    pass
finally:
    dhm.set_led(0, 0, 0)
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(0, 0, 0))
    dhm.display()
    print("\nBye.")
