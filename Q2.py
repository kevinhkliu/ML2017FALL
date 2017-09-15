import sys
from PIL import Image
im=Image.open(sys.argv[1])
pix = im.load()
width, height=im.size
imNew = Image.new("RGB", (width,height), "white")

for x in range(width):
    for y in range(height):
        r, g, b = im.getpixel((x, y))
        r = int(r/2)
        g = int(g/2)
        b = int(b/2)
        imNew.putpixel((x,y), (r,g,b))
imNew.save("Q2.png")

