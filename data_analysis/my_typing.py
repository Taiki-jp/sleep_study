from __future__ import annotations

from collections.abc import Callable
from typing import List, NewType

# 型エイリアス
vector = List[float]
# 異なる方を作成するためにはNetType()ヘルパー関数を使う
user_id = NewType("UserId", int)
# 呼び出し可能オブジェクト
def anync_query(
    on_success: Callable[[int], None],
    on_error: Callable[[int, Exception], None],
) -> None:
    return


def main():
    return


if __name__ == "__main__":
    main()
