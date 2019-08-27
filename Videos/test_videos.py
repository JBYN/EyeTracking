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
    circle = Circle(r, "#000", "obj")
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


class Screen:

    def __init__(self, root: Tk, canvas_width: float, canvas_height: float, circle: Circle, positions: list):
        self.circle = circle
        self.pos = positions
        self.current_pos = self.pos.__getitem__(0)
        self.canvas = self.create_canvas(root, canvas_width, canvas_height)
        self.draw_circle()

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

    def update_screen(self, positions: list):
        self.pos = positions

    def move_circle(self):
        self.pos.__delitem__(0)
        self.set_current_pos(self.pos.__getitem__(0))

    def create_canvas(self, master: Tk, width_canvas: float, height_canvas: float) -> Canvas:
        canvas = Canvas(master, width=width_canvas, height=height_canvas)
        canvas.pack()
        canvas.tag_bind(self.get_circle().get_tag(), "<Button-1>", self.on_click_handler)
        return canvas

    def on_click_handler(self, event):
        i = 0
        t = time.time()
        t2 = time.time()
        while (t2 - t) < 5:
            b, img = CAM.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(gray)
            position.append((self.get_current_pos()[0], self.get_current_pos()[1]))
            t2 = time.time()
            i += 1

        event.widget.delete(event.widget.find_withtag(CURRENT))
        if self.get_positions().__len__() > 1:
            self.move_circle()
            self.draw_circle()
        else:
            self.write_data()

    def draw_circle(self):
        x = self.get_current_pos()[0]
        y = self.get_current_pos()[1]
        radius = self.get_circle().get_radius()
        self.get_canvas().create_oval(x - radius, y - radius, x + radius, y + radius,
                                      fill=self.get_circle().get_color(), tag=self.get_circle().get_tag())

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
