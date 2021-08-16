import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets  import RectangleSelector
from matplotlib.patches import Rectangle
# xdata = np.linspace(0,9*np.pi, num=301)
# ydata = np.sin(xdata)

# fig, ax = plt.subplots()
# ax.plot(xdata, ydata)



def tellme(s):
    print(s)
    plt.title(s, fontsize = 16)
    plt.draw() 
# tellme('left click to started\n  double click to confirm')

class window_motion:
    def  __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.pressed = None
        self.region = []
        self.rect = Rectangle((0,0), 0, 0, fill=False, ec='r')
        self.ax.add_patch(self.rect)
    def onclick(self, event):
        self.pressed = True
        self.double = event.dblclick
        if event.button == 1:
            self.left = True
        elif event.button == 3:
            self.right = True
        self.xcoor = event.xdata
        self.ycoor = event.ydata
        
        
        if self.double:
            self.region.pop()
            plt.close()
            print(self.region)
        
        self.left = None
        self.right = None

    def onrelease(self, event):
        self.pressed = False
        #plot
        # pass
        
        self.region = []
        self.region.append([[self.xcoor, self.ycoor], [event.xdata, event.ydata]])
        
        self.drawrect(event)
        tellme('region {:.2f} <= x <={:.2f}, {:.2f} <= y <= {:.2f}\n double click to confrim'.format(self.xcoor, event.xdata, event.ydata, self.ycoor))
        

    def show(self):
        print('%s click: button=%d, x=%s, y=%s, ' %
          ('double' if self.double else 'single', 1 if self.left else 3,
           str(self.xcoor), str(self.ycoor)))
    def connect(self):
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        rid = self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        mot = self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)

    def onmotion(self, event):
        if self.pressed:
            self.drawrect(event)
            
    def drawrect(self, event):
        self.tmp_x, self.tmp_y = event.xdata, event.ydata
        x_vec = self.tmp_x - self.xcoor
        y_vec = self.tmp_y - self.ycoor
        self.rect.set_width(abs(x_vec))
        self.rect.set_height(abs(y_vec))
        self.rect.set_xy((min(event.xdata, self.xcoor), min(event.ydata, self.ycoor)))
        # self.ax.remove()
        # self.ax.plot([self.xcoor, event.xdata], [self.ycoor, self.ycoor])
        # self.ax.plot([self.xcoor, event.xdata], [event.ydata, event.ydata])
        # self.ax.plot([event.xdata, event.xdata], [self.ycoor, event.ydata])
        # self.ax.plot([self.xcoor, self.xcoor], [self.ycoor, event.ydata])
        self.fig.canvas.draw()
        

        




# wm = window_motion(fig, ax)
# wm.connect()
# plt.show()
