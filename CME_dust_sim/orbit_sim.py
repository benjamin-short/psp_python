#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:38:04 2025

@author: besh2109
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Wedge
from astropy.time import Time
import astropy.units as u
from sunpy.coordinates import get_horizons_coord
from astropy.coordinates import HeliocentricMeanEcliptic
from astropy.constants import G,M_sun,R_sun
import time
from requests.exceptions import HTTPError

# Gravitational constant in AU^3 / (day^2 * M_sun)
G = G.to(u.AU**3/(u.M_sun*u.d**2)).value
# Mass of sun in units of itself. i.e. 1
M_sun = M_sun.to(u.M_sun).value

# Time range
start_time = Time('2022-09-01')
end_time = Time('2022-09-10')

# Force time window
force_start = Time('2022-09-05').jd  # Julian Day
force_end = Time('2022-09-8').jd

# Force angular range in J2000 ecliptic (degrees)
theta_min_force = 90
theta_max_force = 180

# WISPR inner camera FOV (HPLN, degrees)
wispr_fov_min = 13  # Inner edge
wispr_fov_max = 55  # Outer edge
wispr_fov_width = wispr_fov_max - wispr_fov_min  # 40°


# dt = 1 * u.day
dt = 0.05 * u.day
times = Time(np.arange(start_time.jd, end_time.jd, dt.value), format='jd')
dt_days = dt.to(u.day).value
steps = str(len(np.arange(start_time.jd, end_time.jd, dt.value)))

# Real bodies (Horizons IDs or names recognized by sunpy)
real_bodies = {
    'Mercury': '199',
    'Venus': '299',
    'Earth': '399',
    'Mars': '499',
    'Parker Solar Probe': '-96'
}

# Color mapping
colors = {
    'Sun': 'yellow',
    'Mercury': 'darkgrey',
    'Venus': 'orange',
    'Earth': 'blue',
    'Mars': 'red',
    'Parker Solar Probe': 'purple',  # Distinct for Parker
    'Test': 'black'  # Default for test particles
}

# Fetch and transform to J2000 ecliptic
def get_orbit_data(body_id, start, stop, step=steps, retries=3, delay=5):
    for attempt in range(retries):
        try:
            # No observer parameter needed; heliocentric by default
            coord = get_horizons_coord(body_id, {'start': start, 'stop': stop, 'step': step})
            # Transform to J2000 ecliptic frame
            j2000_coord = coord.transform_to(HeliocentricMeanEcliptic(equinox='J2000'))
            x = j2000_coord.cartesian.x.to(u.au).value
            y = j2000_coord.cartesian.y.to(u.au).value
            return x, y
        except HTTPError as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt + 1} for {body_id} failed with {e}, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise Exception(f"Failed to fetch {body_id} after {retries} attempts: {e}")


# Particle class for real bodies (sunpy.coordinates)
class RealParticle:
    def __init__(self, name, body_id):
        self.name = name
        # self.history_x, self.history_y = get_orbit_data(body_id, times)
        self.history_x, self.history_y = get_orbit_data(body_id, start_time, end_time)

class TestParticle:
    def __init__(self, x0, y0, vx0, vy0):
        self.x = x0
        self.y = y0
        self.vx = vx0
        self.vy = vy0
        self.history_x = [x0]
        self.history_y = [y0]
        self.name = 'Test'

    def update(self, current_time):
        r = np.sqrt(self.x**2 + self.y**2)
        # if r < 0.01:
        #     r = 0.01
        # Gravity (always active)
        ax_grav = -G * M_sun * self.x / r**3
        ay_grav = -G * M_sun * self.y / r**3
        
        # Radial force (time and angle dependent)
        ax_radial = ay_radial = 0
        if force_start <= current_time <= force_end:
            # Calculate angle in J2000 ecliptic (degrees)
            theta = np.degrees(np.arctan2(self.y, self.x))
            # Normalize to 0-360°
            if theta < 0:
                theta += 360
            # Apply force if within angular range
            if theta_min_force <= theta <= theta_max_force:
                # force_magnitude = 0.0001 / r**2
                force_magnitude = 0.00037 / r**2
                ax_radial = force_magnitude * self.x / r
                ay_radial = force_magnitude * self.y / r
        
        # Total acceleration
        ax = ax_grav + ax_radial
        ay = ay_grav + ay_radial
        self.vx += ax * dt_days
        self.vy += ay * dt_days
        self.x += self.vx * dt_days
        self.y += self.vy * dt_days
        self.history_x.append(self.x)
        self.history_y.append(self.y)

# Load real bodies
particles = [RealParticle(name, body_id) for name, body_id in real_bodies.items()]
n_steps = len(particles[0].history_x)
times = Time(np.linspace(start_time.jd, end_time.jd, n_steps), format='jd')

    
# Add test particles (randomly distributed in inner Solar System)
n_test = 13  # Increased for visibility
for _ in range(n_test):
    # Random radial distance (0.3 to 2 AU, inner Solar System)
    r = np.random.uniform(0.07, 0.1)
    # r = np.random.uniform(0.3, 0.5)
    theta = np.random.uniform(0, 2*np.pi)
    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)
    # Circular orbit velocity (perturbed slightly)
    v_circ = np.sqrt(G * M_sun / r)
    vx0 = -v_circ * np.sin(theta) * np.random.uniform(0.95, 1.05)  # ±5% perturbation
    vy0 = v_circ * np.cos(theta) * np.random.uniform(0.95, 1.05)
    particles.append(TestParticle(x0, y0, vx0, vy0))

# Simulate test particles
for i in range(1, n_steps):
    current_time = times[i].jd
    for p in particles:
        if isinstance(p, TestParticle):
            p.update(current_time)


# Animation setup
figure_width_inches = 12  # Figure width in inches
dpi = plt.rcParams['figure.dpi']  # Default DPI (typically 100)
plot_width_au = 1.2  # Arbitrary plot width in AU
ax_lim_au = plot_width_au/2

fig, ax = plt.subplots(figsize=(figure_width_inches, figure_width_inches))
ax.set_xlim(-ax_lim_au, ax_lim_au)
ax.set_ylim(-ax_lim_au, ax_lim_au)
ax.set_aspect('equal')
ax.grid(True)

# Sun’s real diameter in AU
sun_diameter_au = 2*R_sun.to(u.AU).value
plot_width_points = figure_width_inches * dpi
sun_diameter_points = (sun_diameter_au / plot_width_au) * plot_width_points
sun_markersize = sun_diameter_points ** 2 / 4  # Exact area-based size
if plot_width_au > 2:
    sun_markersize = 15 #just set it so that its perceptible on larger scales.

sun_marker = ax.plot(0, 0, 'o', color=colors['Sun'], markersize=sun_markersize, label='Sun', alpha=1)

# sun_marker = plt.Circle((0, 0), sun_diameter_au, color=colors['Sun'], label='Sun')
# ax.add_artist(sun_marker)

# Lines and scatters with specific colors
lines = []
scatters = []
test_label_added = False
for p in particles:
    color = colors[p.name]
    if p.name == 'Test' and not test_label_added:
        lines.append(ax.plot([], [], color=color, alpha=0.7, label='Test')[0])
        scatters.append(ax.plot([], [], 'o', color=color, markersize=5)[0])
        test_label_added = True
    else:
        lines.append(ax.plot([], [], color=color, alpha=0.7, label=p.name if p.name != 'Test' else None)[0])
        scatters.append(ax.plot([], [], 'o', color=color, markersize=5)[0])



# Force zone wedge
force_wedge = Wedge((0, 0), ax_lim_au + 0.5, theta_min_force, theta_max_force, color='red', alpha=0.2, label='Force Zone')
ax.add_patch(force_wedge)
force_wedge.set_visible(False)

# WISPR FOV wedge
psp_idx = list(real_bodies.keys()).index('Parker Solar Probe')
wispr_wedge = Wedge((0, 0), 0, 0, 0, color='blue', alpha=0.1, label='WISPR FOV')
ax.add_patch(wispr_wedge)
wispr_wedge.set_visible(False)


# Time display in bottom right
time_text = ax.text(0.95, 0.05, '', transform=ax.transAxes, ha='right', va='bottom', fontsize=10)

trail_length = 150

def init():
    for line, scatter in zip(lines, scatters):
        line.set_data([], [])
        scatter.set_data([], [])
    force_wedge.set_visible(True)
    wispr_wedge.set_visible(True)
    time_text.set_text('')
    return lines + scatters + [force_wedge, wispr_wedge, time_text]

def animate(i):
    current_time = times[i].jd
    if force_start <= current_time <= force_end:
        force_wedge.set_visible(True)
    else:
        force_wedge.set_visible(False)
    
    # Update WISPR FOV from PSP, offset toward ram side
    psp_x = particles[psp_idx].history_x[i]
    psp_y = particles[psp_idx].history_y[i]
    r_psp = np.sqrt(psp_x**2 + psp_y**2)
    # print((r_psp*u.AU).to(u.R_sun))
    sun_theta = np.degrees(np.arctan2(-psp_y, -psp_x))  # PSP to Sun
    if sun_theta < 0:
        sun_theta += 360

    # WISPR FOV 15° to 55° from Sun direction, tilted toward ram side
    wispr_theta_min = sun_theta - wispr_fov_min
    wispr_theta_max = sun_theta - wispr_fov_max
    wispr_wedge.set_center((psp_x, psp_y))
    wispr_wedge.set_radius(0.6)
    wispr_wedge.set_theta1(wispr_theta_max)
    wispr_wedge.set_theta2(wispr_theta_min)
    wispr_wedge.set_visible(True)
    
    
    for j, (p, line, scatter) in enumerate(zip(particles, lines, scatters)):
        start_idx = max(0, i + 1 - trail_length)
        line.set_data(p.history_x[start_idx:i+1], p.history_y[start_idx:i+1])
        scatter.set_data([p.history_x[i]], [p.history_y[i]])
    
    time_str = times[i].strftime('%Y-%m-%d %H:%M UTC')
    time_text.set_text(time_str)
    
    return lines + scatters + [force_wedge, wispr_wedge, time_text]



ani = FuncAnimation(fig, animate, init_func=init, frames=len(times),
                   interval=50, blit=True)
plt.title("CME Force Sim with Test Particles and WISPR FOV, "+str(start_time)[:10]+' - '+str(end_time)[:10])


# Manual legend with flat list of handles
plt.legend()


writer = FFMpegWriter(fps=30, bitrate=1800, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])

save_path = "/Users/besh2109/Desktop/"
save_name = "sunpy_orbit_with_test_particles.mp4"

ani.save(save_path+save_name, writer=writer)
print(f"Video saved as '{save_name}'")