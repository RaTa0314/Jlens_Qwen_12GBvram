"""
fix_unembed.py
================
自動修正 phase4_fit_asr.py 裡的 Qwen2AudioLensModel.unembed()，
把 float32 的 residual cast 回模型實際 dtype 再送進 norm/lm_head。

用正規表達式定位整個 unembed() 函式區塊（從 "def unembed" 開始，
到下一個同縮排層級的 "def " 或檔案結尾為止）整段替換，不靠手動複製貼上，
避免再發生 tab/space 混用或字元被吃掉的問題。

執行：
    python fix_unembed.py
"""

import re
import shutil

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
    backup_path = PATH + ".bak"
    shutil.copyfile(PATH, backup_path)
    print(f"已備份原檔到 {backup_path}")

    with open(PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 找「def unembed(self, residual):」開始，到下一個同樣縮排 4 格的 "def "
    # （也就是同一個 class 裡的下一個 method）或檔案結尾為止，整段換掉。
    pattern = re.compile(
        r"^    def unembed\(self, residual\):\n"
        r"(?:.*\n)*?"                      # 函式內容（非貪婪，跨多行）
        r"(?=^    def |\Z)",                # 直到下一個同縮排的 def，或檔案結尾
        re.MULTILINE,
    )

    new_content, n_subs = pattern.subn(NEW_METHOD, content)

    if n_subs == 0:
        print("[錯誤] 找不到 unembed() 函式，可能它已經被改過或格式不同。")
        print("請貼出目前 phase4_fit_asr.py 裡 unembed() 附近的實際內容，我再另外處理。")
        return
    if n_subs > 1:
        print(f"[警告] 找到 {n_subs} 處符合的區塊，這不太正常，請檢查檔案是否有重複定義。")

    with open(PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"已修正 {n_subs} 處 unembed()，寫回 {PATH}")

    # 立刻驗證檔案至少能被 Python 正確 parse（不會有縮排/語法錯誤）
    import ast
    try:
        ast.parse(new_content)
        print("語法檢查通過。")
    except SyntaxError as e:
        print(f"[錯誤] 修正後的檔案有語法錯誤: {e}")
        print(f"已保留備份在 {backup_path}，可以還原：cp {backup_path} {PATH}")


if __name__ == "__main__":
    main()
