import numpy as np
import plotly
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_3dpc(X, title='reconstruction', point_colors='#3366CC', size=0.5):
    data = []
    data.append(get_3D_scater_trace(X, point_colors, '3D points', size=size))
    fig = go.Figure(data=data)
    path = title + '.html'
    plotly.offline.plot(fig, filename=path, auto_open=False)


def get_3D_scater_trace(points, color, name='3d_points', size=0.5):
    assert points.shape[1] == 3, "3d plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d plot input points are not correctely shaped "

    trace = go.Scatter3d(
        name=name,
        x=points[:,0],
        y=points[:,1],
        z=points[:,2],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
        )
    )
    return trace




def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

class PointsAnimator():
    """
    Create the animator with directory and title.
    Add frames of points
    Finally make animation
    """
    def __init__(self, dir='results',title='animation'):
        self.save_path = os.path.join(dir,'{}.html'.format(title))
        self.frames = []
        self.titles = []
        self.num_frame = 0

    def add_frame(self, pts3D, params=None):
        data = []
        if params and 'ref_points' in params.keys():
            ref_points = params['ref_points']
            data.append(get_3D_scater_trace(ref_points - ref_points.mean(axis=0),
                                            '#990000', 'reference points', size=2))

        data.append(get_3D_scater_trace(pts3D - pts3D.mean(axis=0),
                                        '#3366CC', '3D points', size=2))
        title = 'animation title'
        frame = go.Frame(data=data, layout=go.Layout(title=title), name=self.num_frame)
        self.frames.append(frame)
        self.num_frame += 1

    def make_animation(self):
        sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(self.frames)
                ],
            }
        ]
        fig = go.Figure(
            data=self.frames[0].data,
            layout=go.Layout(
                #xaxis=dict(range=[-10, 10], autorange=False),
                #yaxis=dict(range=[-10, 10], autorange=False),
                # updatemenus=[dict(
                #     type="buttons",
                #     buttons=[dict(label="Play",
                #                   method="animate",
                #                   args=[None])])],
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(50)],
                                "label": "&#9654;",  # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "&#9724;",  # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                ],
                sliders=sliders
            ),
            frames=self.frames
        )
        #print(len(self.frames))

        fig.write_html(self.save_path)
        #plotly.offline.plot(fig, filename=self.save_path)


def two_pc(pc1,pc2):
    fig = plt.figure(tight_layout={'pad':0, 'h_pad':0, 'w_pad':0, 'rect':[0,0,1,1]})
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.axis('off')
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c=np.zeros([1,3]))
    ax = fig.add_subplot(1,2,2,projection='3d')
    ax.axis('off')
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2],c=np.zeros([1,3]))
    plt.show()


def moving_points_video(points,colors,title='animation',sizes=2):
    n_frames = points.shape[0]
    fig = plt.figure(tight_layout={'pad':0, 'h_pad':0, 'w_pad':0, 'rect':[0,0,1,1]})
    ax = fig.add_subplot(projection='3d')

    # im = ax.scatter(points[-1, :, 0], points[-1, :, 1], points[-1, :, 2], c=colors, s=sizes)
    ax.view_init(azim=90, elev=-70)
    ax.axis('off')
    # plt.show()
    frames = []
    for i in np.arange(0, 1000, 5):
        #print(i)
        im = ax.scatter(points[i,:, 0], points[i,:, 1], points[i,:, 2], c=colors, s=sizes)
        frames.append([im])

    ani = animation.ArtistAnimation(fig, frames)
    print('before save')
    #plt.show()
    ani.save('{}.mp4'.format(title), fps=30, extra_args=['-vcodec', 'libx264'])
    print('after save')


def static_points_video(points, colors, n_frames=1, title='animation', sizes=2):
    """

    :param points: [N,3]
    :param colors: [N,3] or [3,]
    :return:
    """
    fig = plt.figure(tight_layout={'pad':0, 'h_pad':0, 'w_pad':0, 'rect':[0,0,1,1]})
    ax = fig.add_subplot(projection='3d')
    ax.axis('off')
    azimuts = np.linspace(start=0, stop=360, num=n_frames)
    elevs = np.linspace(start=-60-90, stop=300-90, num=n_frames)
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes)
    # ax.view_init(elev=90, azim=-60)
    # plt.show()
    def init():
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes)
        return fig,

    def animate(i):
        ax.view_init(elev=elevs[i], azim=azimuts[i])
        return fig,

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=10, blit=True)
    # Save
    anim.save(title+'.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
    plt.close()



if __name__ == "__main__":
    points = np.random.rand(10,30,3)
    moving_points_video(points, np.zeros([1, 3]), title='animation', sizes=2)
