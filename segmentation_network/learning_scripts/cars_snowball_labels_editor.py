import tkinter
from PIL import ImageTk, Image
import numpy as np
import copy
from math import sqrt

from segmentation_network.constants import INPUT_SIZE


class LabelsEditor(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def update(self):
        img_with_filter = copy.deepcopy(self.img)
        for x in range(INPUT_SIZE[0]):
            for y in range(INPUT_SIZE[1]):
                if self.labels[0, x, y, 0]:
                    for z in range(3):
                        img_with_filter[x, y, z] = 0

        self.root.tk_img = ImageTk.PhotoImage(Image.fromarray(img_with_filter, mode='RGB'))
        self.c.create_image((0, 0), anchor='nw', image=self.root.tk_img)

    def __init__(self, img, labels):
        self.img = img
        self.labels = labels

        self.root = tkinter.Tk()

        self.pen_button = tkinter.Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.eraser_button = tkinter.Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=1)

        self.choose_size_button = tkinter.Scale(self.root, from_=1, to=50, orient=tkinter.HORIZONTAL)
        self.choose_size_button.grid(row=0, column=2)
        self.c = tkinter.Canvas(self.root, bg='white', width=INPUT_SIZE[0], height=INPUT_SIZE[1])
        self.update()

        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=tkinter.RAISED)
        some_button.config(relief=tkinter.SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()

        for x in range(INPUT_SIZE[0]):
            for y in range(INPUT_SIZE[1]):
                if sqrt((x - event.x) ** 2 + (y - event.y) ** 2) < self.line_width:
                    self.labels[0, y, x, 0] = 0. if self.eraser_on else 1.

        self.update()

    def reset(self, event):
        self.old_x, self.old_y = None, None
