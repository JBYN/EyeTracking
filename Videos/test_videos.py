from tkinter import *
import cv2
import time

global r
r = 5

CAM = cv2.VideoCapture(0)
images = list()
position = list()


def main():
    root = Tk()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    config_main_window(root, w, h, "Main Window")
    circle = Circle(r, "red", "calibrate")
    pos = [(w/6, h/6), (5*w/6, 5*h/6), (5*w/6, h/2), (w/2, h/6), (w/2, 5*h/6), (w/6, h/2), (5*w/6, h/6), (w/6, 5*h/6),
           (w/2, h/2)]
    screen = Screen(root, w, h, circle, pos)
    root.mainloop()


def config_main_window(master, width: float, height: float, title: str):
    master.geometry("{}x{}".format(width, height))
    master.title(title)


class Circle:

    def __init__(self, radius: float, color: str, tag: str):
        self.r = radius
        self.color = color
        self.tag = tag

    def get_radius(self) -> float:
        return self.r

    def get_color(self) -> str:
        return self.color

    def get_tag(self) -> str:
        return self.tag

    def set_radius(self, radius: float):
        self.r = radius

    def set_color(self, color: str):
        self.color = color

    def set_tag(self, tag: str):
        self.tag = tag

    def update(self, radius=None, color=None, tag=None):
        if radius is not None:
            self.set_radius(radius)
        if color is not None:
            self.set_color(color)
        if tag is not None:
            self.set_tag(tag)


class Screen:

    def __init__(self, root: Tk, canvas_width: float, canvas_height: float, circle: Circle, positions: list):
        self.root = root
        self.circle = circle
        self.pos = positions
        self.current_pos = self.pos.__getitem__(0)
        self.canvas = self.create_canvas(root, canvas_width, canvas_height)
        self.draw_circle()

    def get_root(self) -> Tk:
        return self.root

    def get_canvas(self) -> Canvas:
        return self.canvas

    def get_circle(self) -> Circle:
        return self.circle

    def get_positions(self) -> list:
        return self.pos

    def get_current_pos(self) -> (float, float):
        return self.current_pos

    def set_current_pos(self, coord: (float, float)):
        self.current_pos = coord

    def set_pos(self, positions: list):
        self.pos = positions

    def update_screen(self, positions: list):
        self.set_pos(positions)
        self.set_current_pos(self.get_positions().__getitem__(0))
        self.update_canvas()
        self.draw_circle()

    def move_circle(self):
        self.pos.__delitem__(0)
        self.set_current_pos(self.pos.__getitem__(0))

    def create_canvas(self, master: Tk, width_canvas: float, height_canvas: float) -> Canvas:
        canvas = Canvas(master, width=width_canvas, height=height_canvas, bg="white")
        canvas.pack()
        canvas.tag_bind(self.get_circle().get_tag(), "<Button-1>", self.on_click_handler)
        return canvas

    def update_canvas(self):
        self.get_canvas().tag_bind(self.get_circle().get_tag(), "<Button-1>", self.on_click_handler)

    def change_color_circle(self, event, color):
        event.widget.delete(self.get_circle().get_tag())
        self.get_circle().update(color=color)
        self.draw_circle()
        self.get_root().update()

    def on_click_handler(self, event):
        i = 0
        t = time.time()
        self.change_color_circle(event, "black")
        t2 = time.time()
        while (t2 - t) < 5:
            b, img = CAM.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(gray)
            position.append((self.get_current_pos()[0], self.get_current_pos()[1]))
            t2 = time.time()
            i += 1

        if self.get_positions().__len__() > 1:
            self.move_circle()
            self.change_color_circle(event, "red")
        elif self.get_circle().get_tag() == "calibrate":
            position.append("End of calibration \n")
            self.get_circle().update(tag="test")
            w = event.widget.winfo_screenwidth()
            h = event.widget.winfo_screenheight()
            pos = [(w/8, h/8), (3*w/8, 5*h/8), (5*w/8, 3*h/8), (7*w/8, 7*h/8), (7*w/8, h/8), (5*w/8, 5*h/8),
                   (3*w/8, 3*h/8), (w/8, 7*h/8), (5*w/8, 7*h/8), (w/8, 3*h/8), (3*w/8, h/8), (7*w/8, 5*h/8),
                   (3*w/8, 7*h/8), (w/8, 5*h/8), (5*w/8, h/8), (7*w/8, 3*h/8)]
            self.update_screen(pos)
        else:
            # self.write_data()
            self.get_root().destroy()

    def draw_circle(self):
        x = self.get_current_pos()[0]
        y = self.get_current_pos()[1]
        radius = self.get_circle().get_radius()
        self.get_canvas().create_oval(x - radius, y - radius, x + radius, y + radius,
                                      fill=self.get_circle().get_color(), tag=self.get_circle().get_tag(),
                                      outline=self.get_circle().get_color())

    @staticmethod
    def write_data():
        index = 0
        f = open("Videos/pos.csv", "w+")
        f.write("X;Y \n")
        for im in images:
            f.write(str(position.__getitem__(index)[0]) + ";" + str(position.__getitem__(index)[1]) + "\n")
            cv2.imwrite("Videos/Images/im_" + str(index) + ".jpg", im)
            index += 1
        f.close()


if __name__ == "__main__":
    main()
