# Author: Suvansh Sanjeev (suvansh@berkeley.edu)
# Course: EECS 127 (UC Berkeley)
# Notes: Parts adapted from http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from scipy.optimize import minimize, OptimizeResult


class Answer:
    def __init__(self, methods, func, grad):
        self.methods = methods
        self.func = func
        self.grad = grad
        self.has_set = self.has_set_fn = False
    
    """ For your use """
    
    def set_fn_settings(self, fn_name):
        self.fn_name = fn_name
        self.xmin, self.xmax, self.xstep = self.get_coord_bounds(fn_name)
        self.ymin, self.ymax, self.ystep = self.get_coord_bounds(fn_name)
        self.x, self.y = np.meshgrid(np.arange(self.xmin, self.xmax + self.xstep, self.xstep),
                                     np.arange(self.ymin, self.ymax + self.ystep, self.ystep))
        self.f = self.get_fg(fn_name)
        self.z = self.f((self.x, self.y))[0]
        self.minima_ = self.get_minimum(fn_name)
        self.elev, self.azim = self.get_elev_azim(fn_name)
        self.has_set_fn = True
    
    def set_settings(self, fn_name, method, x0, **kwargs):
        if method not in self.methods:
            raise ValueError('Invalid method %s' % method)
        self.set_fn_settings(fn_name)
        self.method = self.methods[method]
        self.x0 = x0
        self.options = kwargs
        path_ = [x0]
        result = minimize(self.f, x0=x0, method=self.method,
                               jac=True, tol=1e-20, callback=self.make_minimize_cb(path_),
                               options=kwargs)
        assert len(result) == 2 and isinstance(result[0], OptimizeResult) and isinstance(result[1], np.ndarray)
        self.res, self.losses = result
        self.path = np.array(path_).T
        self.has_set = True
    
    def get_settings(self):
        return self.fn_name, self.method.__name__, self.x0, self.options
    
    def compare(self, method, start_iter=0, **kwargs):
        res1, losses1 = self.res, self.losses
        curr_settings = self.get_settings()
        self.set_settings(self.fn_name, method, self.x0, **kwargs)
        res2, losses2 = self.res, self.losses
        # plot training curves
        method1 = curr_settings[1]
        method2 = self.method.__name__
        plt.plot(np.arange(len(losses1)-start_iter), losses1[start_iter:], label=method1)
        plt.plot(np.arange(len(losses2)-start_iter), losses2[start_iter:], label=method2)
        plt.title('Training Curve')
        plt.legend()
        plt.show()
        print('[Method {:>10}] Final loss: {:.4f}, Final x: [{:.4f}, {:.4f}]'.format(method1, losses1[-1], res1.x[0], res1.x[1]))
        print('[Method {:>10}] Final loss: {:.4f}, Final x: [{:.4f}, {:.4f}]'.format(method2, losses2[-1], res2.x[0], res2.x[1]))
        self.set_settings(*curr_settings[:-1], **curr_settings[-1])
        
    def plot2d(self):
        self.check_set_fn()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contour(self.x, self.y, self.z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
        ax.plot(*self.minima_, 'r*', markersize=18)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        plt.show()
    
    def plot3d(self):
        self.check_set_fn()
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection='3d', elev=self.elev, azim=self.azim)
        ax.plot_surface(self.x, self.y, self.z, norm=LogNorm(), rstride=1, cstride=1, 
                        edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        ax.plot(*self.minima_, self.f(self.minima_)[0], 'r*', markersize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(self.method.__name__)
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        plt.show()
        
    def path2d(self):
        self.check_set()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contour(self.x, self.y, self.z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
        ax.quiver(self.path[0,:-1], self.path[1,:-1], self.path[0,1:]-self.path[0,:-1], self.path[1,1:]-self.path[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
        ax.plot(*self.minima_, 'r*', markersize=18)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        plt.show()
        
    def path3d(self):
        self.check_set()
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection='3d', elev=self.elev, azim=self.azim)

        ax.plot_surface(self.x, self.y, self.z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        ax.quiver(self.path[0,:-1], self.path[1,:-1], self.f(self.path[::,:-1])[0], 
                  self.path[0,1:]-self.path[0,:-1], self.path[1,1:]-self.path[1,:-1],
                  self.f(self.path[::,1:])[0]-self.f(self.path[::,:-1])[0], 
                  normalize=True, color='k')
        ax.plot(*self.minima_, self.f(self.minima_)[0], 'r*', markersize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(self.method.__name__)
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        plt.show()
        
    def video2d(self):
        self.check_set()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contour(self.x, self.y, self.z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
        ax.plot(*self.minima_, 'r*', markersize=18)
        line, = ax.plot([], [], 'b', label=self.method.__name__, lw=2)
        point, = ax.plot([], [], 'bo')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        ax.legend(loc='upper left')
        anim = animation.FuncAnimation(fig, self.get_animate2d(line, point), init_func=self.get_init2d(line, point),
                                       frames=self.path.shape[1], interval=60, 
                                       repeat_delay=5, blit=True)
        filename = filename or '%s_%s_2d.mp4' % (self.fn_name, self.method.__name__)
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        anim.save(filename, writer=animation.writers['imagemagick'](fps=15))
        
    def video3d(self, filename=None):
        self.check_set()
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection='3d', elev=self.elev, azim=self.azim)
        ax.plot_surface(self.x, self.y, self.z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        ax.plot(*self.minima_, self.f(self.minima_)[0], 'r*', markersize=10)
        line, = ax.plot([], [], [], 'b', label=self.method.__name__, lw=2)
        point, = ax.plot([], [], [], 'bo')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(self.method.__name__)
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        anim = animation.FuncAnimation(fig, self.get_animate3d(line, point), init_func=self.get_init3d(line, point),
                                       frames=self.path.shape[1], interval=60, 
                                       repeat_delay=5, blit=True)
        filename = filename or '%s_%s_3d.mp4' % (self.fn_name, self.method.__name__)
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        anim.save(filename, writer=animation.writers['imagemagick'](fps=15))
        
    def get_xs_losses(self):
        self.check_set()
        return self.path.T, self.losses
    
    def get_min_errs(self):
        """ Returns the best x differences and function differences over the run. """
        x_err = np.linalg.norm(self.path - self.minima_, axis=0).min()
        loss_err = (self.losses - self.f(self.minima_)[0]).min()
        return x_err, loss_err
    
    def func_val(self, x):
        return self.f(x)[0]
    
    def grad_val(self, x):
        return self.f(x)[1]

    """ Under the hood """
    
    def check_set_fn(self):
        assert self.has_set_fn, "Need to call `set_fn_settings` first."
    
    def check_set(self):
        assert self.has_set, "Need to call `set_settings` first."
        
    def get_fg(self, fn_name):
        return lambda x: (self.func(fn_name, x[0], x[1]), self.grad(fn_name, x[0], x[1]))
    
    def get_coord_bounds(self, fn):
        if fn == 'booth':
            return -10, 10, 0.4
        elif fn == 'beale':
            return -4.5, 4.5, 0.2
        elif fn == 'rosen2d':
            return -5, 10, 0.3
        elif fn == 'ackley2d':
            return -32.768, 32.768, 0.8192
        else:
            raise ValueError('Invalid function %s' % fn)
    
    def get_minimum(self, fn):
        if fn == 'booth':
            return np.array([1., 3.]).reshape(-1, 1)
        elif fn == 'beale':
            return np.array([3., .5]).reshape(-1, 1)
        elif fn == 'rosen2d':
            return np.array([1., 1.]).reshape(-1, 1)
        elif fn == 'ackley2d':
            return np.array([0., 0.]).reshape(-1, 1)
        else:
            raise ValueError('Invalid function %s' % fn)

    def get_elev_azim(self, fn):
        if fn == 'booth':
            return 30, -50
        elif fn == 'beale':
            return 50, -140
        elif fn == 'rosen2d':
            return 40, 140
        elif fn == 'ackley2d':
            return 30, 40
        else:
            raise ValueError('Invalid function %s' % fn)
        
    def make_minimize_cb(self, path=[]):
        return lambda xk: path.append(np.copy(xk))
    
    def get_init2d(self, line, point):
        def init2d():
            line.set_data([], [])
            point.set_data([], [])
            return line, point
        return init2d
    
    def get_animate2d(self, line, point):
        def animate2d(i):
            line.set_data(*self.path[::,:i])
            point.set_data(*self.path[::,i-1:i])
            return line, point
        return animate2d
        
    def get_init3d(self, line, point):
        def init3d():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        return init3d

    def get_animate3d(self, line, point):
        def animate3d(i):
            line.set_data(self.path[0,:i], self.path[1,:i])
            line.set_3d_properties(self.f(self.path[::,:i])[0])
            point.set_data(self.path[0,i-1:i], self.path[1,i-1:i])
            point.set_3d_properties(self.f(self.path[::,i-1:i])[0])
            return line, point
        return animate3d
