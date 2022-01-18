import glob
import os
import sys

from pptx import Presentation
from pptx.util import Inches

fig_dir = glob.glob("c:/users/taiki/desktop/figures/*.png")

prs = Presentation("c:/users/taiki/desktop/template.pptx")

title_slide_layout = prs.slide_layouts[7]
for counter, img_path in enumerate(fig_dir):
    for i in range(9):
        i = counter * 9 + i
        slide = prs.slides.add_slide(title_slide_layout)
        left = top = Inches(1)
        pic = slide.shapes.add_picture(img_path, left, top)

prs.save("c:/users/taiki/desktop/test.pptx")
