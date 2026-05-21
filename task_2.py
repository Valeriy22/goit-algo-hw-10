"""
Завдання 2: Обчислення визначеного інтеграла методом Монте-Карло.

Інтегруємо функцію f(x) = x^2 на проміжку [0, 2].
Аналітично: ∫₀² x² dx = x³/3 |₀² = 8/3 ≈ 2.6666666...

Порівнюємо результат із функцією scipy.integrate.quad.
"""

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt


# Функція для інтегрування
def f(x):
    return x ** 2


# Межі інтегрування
A = 0  # Нижня межа
B = 2  # Верхня межа


def monte_carlo_integration(func, a: float, b: float, n: int = 1_000_000) -> float:
    """
    Обчислення визначеного інтеграла методом Монте-Карло (метод середнього значення).

    Ідея: ∫ₐᵇ f(x) dx ≈ (b - a) * <f(x)>, де <f(x)> — середнє значення функції
    на проміжку [a, b], яке оцінюємо як середнє арифметичне f у випадкових точках.

    :param func: функція, інтеграл якої обчислюється.
    :param a: нижня межа інтегрування.
    :param b: верхня межа інтегрування.
    :param n: кількість випадкових точок.
    :return: наближене значення інтеграла.
    """
    x_random = np.random.uniform(a, b, n)
    y_values = func(x_random)
    return (b - a) * np.mean(y_values)


def monte_carlo_geometric(func, a: float, b: float, n: int = 1_000_000) -> float:
    """
    Обчислення визначеного інтеграла геометричним методом Монте-Карло
    (метод "влучань" — hit-or-miss).

    Ідея: генеруємо випадкові точки у прямокутнику [a, b] × [0, y_max]
    і рахуємо частку точок під кривою. Площа під кривою наближено дорівнює
    добутку площі прямокутника на цю частку.

    :param func: функція (повинна бути невід'ємною на [a, b]).
    :param a: нижня межа інтегрування.
    :param b: верхня межа інтегрування.
    :param n: кількість випадкових точок.
    :return: наближене значення інтеграла.
    """
    # Знаходимо максимум функції на [a, b] (для f(x)=x^2 на [0,2] це f(2)=4)
    x_grid = np.linspace(a, b, 1000)
    y_max = np.max(func(x_grid))

    # Генеруємо випадкові точки в прямокутнику
    x_random = np.random.uniform(a, b, n)
    y_random = np.random.uniform(0, y_max, n)

    # Рахуємо кількість точок, які потрапили під криву
    under_curve = np.sum(y_random <= func(x_random))

    # Площа = (площа прямокутника) * (частка точок під кривою)
    rectangle_area = (b - a) * y_max
    return rectangle_area * under_curve / n


def plot_function() -> None:
    """Будує графік функції з виділеною областю інтегрування."""
    x = np.linspace(-0.5, 2.5, 400)
    y = f(x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "r", linewidth=2, label=r"$f(x) = x^2$")

    # Заповнення області під кривою
    ix = np.linspace(A, B, 200)
    iy = f(ix)
    ax.fill_between(ix, iy, color="gray", alpha=0.3, label="Область інтегрування")

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, max(y) + 0.1])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.axvline(x=A, color="gray", linestyle="--")
    ax.axvline(x=B, color="gray", linestyle="--")
    ax.set_title(f"Графік інтегрування f(x) = x² від {A} до {B}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("integration_plot.png", dpi=100)
    print("Графік збережено у файл integration_plot.png")


def analyze_convergence() -> None:
    """Аналізує збіжність методу Монте-Карло при збільшенні кількості точок."""
    exact, _ = spi.quad(f, A, B)
    sample_sizes = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    print()
    print("Залежність точності методу Монте-Карло від кількості точок:")
    print("=" * 78)
    print(
        f"{'N точок':>12} | {'Mean (середнє)':>17} | {'Hit-or-miss':>14} | "
        f"{'Помилка (mean)':>17}"
    )
    print("-" * 78)
    for n in sample_sizes:
        mc_mean = monte_carlo_integration(f, A, B, n)
        mc_hit = monte_carlo_geometric(f, A, B, n)
        error = abs(mc_mean - exact)
        print(
            f"{n:>12,} | {mc_mean:>17.10f} | {mc_hit:>14.6f} | {error:>17.10f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    # Фіксуємо seed для відтворюваності
    np.random.seed(42)

    # 1. Будуємо графік
    plot_function()
    print()

    # 2. Обчислюємо інтеграл різними способами
    n_points = 1_000_000

    mc_mean = monte_carlo_integration(f, A, B, n_points)
    mc_hit = monte_carlo_geometric(f, A, B, n_points)
    quad_result, quad_error = spi.quad(f, A, B)
    analytical = (B ** 3 - A ** 3) / 3  # ∫ x² dx = x³/3

    print("Результати обчислення інтеграла ∫₀² x² dx:")
    print("-" * 56)
    print(f"Аналітичне значення (8/3):            {analytical:.10f}")
    print(f"scipy.integrate.quad:                 {quad_result:.10f}")
    print(f"  абсолютна похибка quad:             {quad_error:.2e}")
    print(f"Монте-Карло (метод середнього):       {mc_mean:.10f}")
    print(f"  різниця з quad:                     {abs(mc_mean - quad_result):.10f}")
    print(f"Монте-Карло (геометричний/hit-or-miss): {mc_hit:.10f}")
    print(f"  різниця з quad:                     {abs(mc_hit - quad_result):.10f}")

    # 3. Аналіз збіжності
    analyze_convergence()
