"""
fix_unembed_v2.py
====================
上一版 fix_unembed.py 的 regex 停止條件只認「同縮排的 def」，沒有把
「檔案裡縮排回到 column 0 的任何內容」（模組層級的函式、註解區塊分隔線）
當成停止點，導致它把 unembed() 之後、直到檔案結尾的所有內容
（load_jsonl_lines、filter_by_length、main()...）都吞掉了。

這一版把停止條件加上 `^\\S`（行首出現非空白字元），也就是只要看到
縮排「歸零」的那一行（不管是註解 # 開頭，還是 def 開頭），就停止匹配。

執行前，請先確認 phase4_fit_asr.py 已經是還原備份後的乾淨版本：
    cp phase4_fit_asr.py.bak phase4_fit_asr.py
    python fix_unembed_v2.py
"""

import re
import shutil
import ast

PATH = "phase4_fit_asr.py"

NEW_METHOD = '''    def unembed(self, residual):
        # transport()/select() 回傳的 residual 是 float32（JacobianLens 為了精度
        # 刻意這麼做），但 norm/lm_head 的權重是 bf16，dtype 對不上會直接報錯，
        # 所以這裡先 cast 回模型實際在用的 dtype 再送進去。
        target_dtype = self._inner.language_model.norm.weight.dtype
        normed = self._inner.language_model.norm(residual.to(target_dtype))
        return self._model.lm_head(normed)
'''


def main():
    with open(PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 防呆：如果檔案裡已經沒有 load_jsonl_lines，代表還沒還原備份，先擋下來
    if "def load_jsonl_lines" not in content:
        print("[錯誤] 目前檔案裡沒有 load_jsonl_lines，代表還沒還原備份。")
        print("請先執行: cp phase4_fit_asr.py.bak phase4_fit_asr.py")
        return

    backup_path = PATH + ".bak2"
    shutil.copyfile(PATH, backup_path)
    print(f"已備份目前版本到 {backup_path}")

    # 停止條件：下一個同縮排 4 格的 def，或是行首「縮排歸零」的任何內容
    # （模組層級的 def / 註解分隔線 / class 定義...），或檔案結尾。
    pattern = re.compile(
        r"^    def unembed\(self, residual\):\n"
        r"(?:.*\n)*?"
        r"(?=^    def |^\S|\Z)",
        re.MULTILINE,
    )

    new_content, n_subs = pattern.subn(NEW_METHOD, content)

    if n_subs != 1:
        print(f"[錯誤] 預期替換 1 處，實際替換了 {n_subs} 處，為求安全不寫入檔案。")
        print("請把目前 phase4_fit_asr.py 裡 unembed() 附近（含前後各 5 行）貼給我人工處理。")
        return

    for required in ["def load_jsonl_lines", "def filter_by_length", "def main"]:
        if required not in new_content:
            print(f"[錯誤] 替換後發現 {required} 不見了，為求安全不寫入檔案。")
            return

    try:
        ast.parse(new_content)
    except SyntaxError as e:
        print(f"[錯誤] 替換後語法有誤: {e}，為求安全不寫入檔案。")
        return

    with open(PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("修正成功：unembed() 已更新，其餘函式（load_jsonl_lines / filter_by_length / main）都還在，語法檢查通過。")


if __name__ == "__main__":
    main()
