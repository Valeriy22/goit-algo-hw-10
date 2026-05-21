"""
Завдання 1: Жадібні алгоритми та динамічне програмування.

Реалізація двох функцій для касової системи, яка видає решту покупцеві:
- find_coins_greedy: жадібний алгоритм
- find_min_coins: алгоритм динамічного програмування

Порівняння часу виконання обох алгоритмів.
"""

import time

COINS = [50, 25, 10, 5, 2, 1]


def find_coins_greedy(amount: int, coins: list[int] = COINS) -> dict[int, int]:
    """
    Жадібний алгоритм: на кожному кроці обираємо найбільший доступний номінал.

    Складність: O(n), де n — кількість номіналів монет.

    :param amount: сума, яку потрібно видати.
    :param coins: список доступних номіналів (за замовчуванням COINS).
    :return: словник {номінал: кількість}.
    """
    result: dict[int, int] = {}
    # На випадок, якщо список номіналів не відсортований
    for coin in sorted(coins, reverse=True):
        if amount <= 0:
            break
        if amount >= coin:
            count, amount = divmod(amount, coin)
            result[coin] = count
    return result


def find_min_coins(amount: int, coins: list[int] = COINS) -> dict[int, int]:
    """
    Алгоритм динамічного програмування: знаходить мінімальну кількість монет
    для формування заданої суми.

    Складність: O(amount * n), де n — кількість номіналів монет,
    amount — задана сума.

    :param amount: сума, яку потрібно видати.
    :param coins: список доступних номіналів (за замовчуванням COINS).
    :return: словник {номінал: кількість}.
    """
    if amount <= 0:
        return {}

    # dp[i] — мінімальна кількість монет, яка дає суму i
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0

    # coin_used[i] — номінал монети, який використано на останньому кроці
    # для отримання оптимального dp[i]
    coin_used = [0] * (amount + 1)

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                coin_used[i] = coin

    # Відновлення розв'язку: послідовно віднімаємо монети, які запам'ятали
    result: dict[int, int] = {}
    remaining = amount
    while remaining > 0:
        coin = coin_used[remaining]
        result[coin] = result.get(coin, 0) + 1
        remaining -= coin

    # Сортуємо словник за зростанням номіналу (як у прикладі з завдання)
    return dict(sorted(result.items()))


def measure_time(func, *args, repeats: int = 5) -> float:
    """
    Допоміжна функція для вимірювання середнього часу виконання.
    Повертає середній час у секундах за `repeats` запусків.
    """
    total = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        total += time.perf_counter() - start
    return total / repeats


def compare_algorithms() -> None:
    """Порівнює час виконання обох алгоритмів на різних сумах."""
    test_amounts = [113, 1_000, 10_000, 100_000, 1_000_000]

    print("=" * 72)
    print(f"{'Сума':>10} | {'Greedy (с)':>14} | {'DP (с)':>14} | {'DP / Greedy':>14}")
    print("-" * 72)

    for amount in test_amounts:
        greedy_time = measure_time(find_coins_greedy, amount)
        dp_time = measure_time(find_min_coins, amount, repeats=1)
        ratio = dp_time / greedy_time if greedy_time > 0 else float("inf")
        print(
            f"{amount:>10} | {greedy_time:>14.8f} | {dp_time:>14.8f} | {ratio:>14.1f}x"
        )

    print("=" * 72)


if __name__ == "__main__":
    # Демонстрація роботи на прикладі з умови задачі (сума 113)
    amount = 113
    print(f"Сума для видачі решти: {amount}")
    print(f"Жадібний алгоритм:          {find_coins_greedy(amount)}")
    print(f"Динамічне програмування:    {find_min_coins(amount)}")
    print()

    # Ще кілька прикладів
    for sample in (1, 7, 99, 350):
        print(
            f"Сума = {sample:>4}: "
            f"greedy = {find_coins_greedy(sample)}, "
            f"dp = {find_min_coins(sample)}"
        )
    print()

    # Порівняння часу виконання
    print("Порівняння часу виконання алгоритмів:")
    compare_algorithms()
