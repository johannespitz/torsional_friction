import numpy as np
import matplotlib.pyplot as plt


def compute_torque(radius, external_force, friction):
    # Effective radius (via stiffness)
    stiffness = 500.0  # N/m
    penetration = external_force / stiffness
    effective_radius = np.sqrt(radius**2 - (radius - penetration) ** 2)

    E_star = 0.75 * external_force * radius ** 0.5 * penetration ** 1.5

    # # Effective radius (Herztian Stress)
    # # https://en.wikipedia.org/wiki/Contact_mechanics
    # # https://physics.stackexchange.com/questions/289543/pressure-of-a-sphere-against-the-ground
    # ν_sphere = 0.3
    # ν_plane = 0.3
    # E_sphere = 1e9
    # E_plane = 1e9
    # effective_radius = np.sqrt( # TODO: Needs to be thrid root
    #     0.75
    #     * external_force
    #     * radius
    #     * (1 - ν_sphere**2 / E_sphere + ν_plane**2 / E_plane)
    # )
    max_pressure = 3 * external_force / (2 * np.pi * effective_radius**2)

    # Sample point on circle
    num_samples = 100000
    length = np.sqrt(np.random.uniform(0, 1, num_samples)) * effective_radius
    angle = np.pi * np.random.uniform(0, 0.5, num_samples)
    x = length * np.cos(angle)
    y = length * np.sin(angle)
    points = np.stack([x, y, np.zeros(num_samples)]).transpose()
    # plt.plot(x, y, 'x')
    # plt.show()

    effective_pressure = max_pressure * (1 - (length**2 / effective_radius**2)) ** 0.5
    # plt.plot(length, effective_pressure, 'x')
    # plt.show()

    # tangential_force = effective_pressure * friction
    # effective_force = np.array(
    #     [
    #         tangential_force,
    #         tangential_force,
    #         effective_pressure,
    #     ]
    # ).transpose()
    # τ = np.abs(np.cross(points, effective_force))

    τ = np.array(
        [
            np.zeros(num_samples),
            np.zeros(num_samples),
            length * effective_pressure * friction,
        ]
    ).transpose()

    total_pressure = np.mean(effective_pressure)
    area = np.pi * effective_radius**2

    total_force = total_pressure * area
    τ = τ * area

    assert (total_force - external_force) / external_force < 0.05

    return penetration, effective_radius, τ


radius = 0.015  # m
external_force = 2.0  # N
friction = 0.8

tau_z = []
pens = []
rads = []
radiuses = np.linspace(0.01, 1.0, 100)
for radius in radiuses:
    penetration, effective_radius, τ = compute_torque(radius, external_force, friction)
    pens.append(penetration)
    rads.append(effective_radius)
    tau_z.append(τ.mean(axis=0)[2])
# plt.plot(radiuses, tau_z)
# plt.plot(radiuses, pens)
plt.plot(radiuses, rads)
plt.show()


radius = 0.015  # m
external_force = 2.0  # N
friction = 0.8

tau_z = []
pens = []
rads = []
forces = np.linspace(0.1, 5, 50)
for external_force in forces:
    penetration, effective_radius, τ = compute_torque(radius, external_force, friction)
    pens.append(penetration)
    rads.append(effective_radius)
    tau_z.append(τ.mean(axis=0)[2])

# plt.plot(forces, pens)
# plt.plot(forces, rads)
plt.plot(forces, tau_z)
plt.show()

radius = 0.015  # m
external_force = 2.0  # N
friction = 0.8

penetration, effective_radius, τ = compute_torque(radius, external_force, friction)

print()
print(f"Penetration:            {penetration:10.4f} m")
print(f"Effective radius:       {effective_radius:10.4f} m")
print(f"Lateral friction force: {external_force * friction:10.4f} N")
print(f"Max Torque: {τ.mean(axis=0)} N")
print()
