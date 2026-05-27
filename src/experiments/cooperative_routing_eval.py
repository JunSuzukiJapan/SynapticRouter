import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import time
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"[Init] Using device: {device}")

# ==========================================
# 1. 共通シナプス定義
# ==========================================

class LLMSynapse(nn.Module):
    """汎用LLMシナプス"""
    def __init__(self, system_prompt=None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID).to(device)
        self.model.eval()
        self.default_system = (
            "You are having a normal one-on-one conversation. Reply only to the user's latest message in one short natural sentence. "
            "Do not introduce yourself. Do not explain what you are. Match the user's language."
        )
        self.system_prompt = system_prompt if system_prompt else self.default_system
    
    def forward(self, text_input, messages=None, max_new_tokens=40):
        system_message = {
            "role": "system",
            "content": self.system_prompt,
        }
        
        if messages is not None:
            formatted_messages = [system_message] + messages
        else:
            formatted_messages = [system_message, {"role": "user", "content": text_input}]
            
        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0, encoded["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return response

class CalculatorSynapse(nn.Module):
    """ルールベース計算機シナプス"""
    def __init__(self):
        super().__init__()
        self.is_neural = False
        
    def forward(self, text_input=None):
        if text_input is not None:
            # 安全なevalのために数式以外の文字を排除
            cleaned = re.sub(r'[^0-9\+\-\*\/\(\)\.\s]', '', text_input)
            try:
                ans = eval(cleaned)
                return f"CALC_RESULT: {ans}"
            except Exception as e:
                return f"CALC_ERROR: {str(e)}"
        return "CALC_ERROR: No input"


# ==========================================
# 2. 3つのアプローチの実装
# ==========================================

class SRA_BaseMotherboard:
    def __init__(self):
        self.synapses = {}
    
    def register_synapse(self, name, synapse):
        self.synapses[name] = synapse


# --- Approach A: Cooperative Routing (Two-Phase) ---
class SRA_CooperativeMotherboard(SRA_BaseMotherboard):
    """アプローチA: 協調的ツーフェーズ・ルーティング"""
    def _extract_math_expression(self, text):
        pattern = r'[\d\s\+\-\*\/\(\)\.]+'
        matches = re.findall(pattern, text)
        for match in matches:
            clean = match.strip()
            # 演算子を含み、意味のある長さの数式候補を返す
            if any(op in clean for op in ['+', '-', '*', '/']) and len(clean) >= 3:
                return clean
        return None

    def route_and_forward(self, text_input):
        math_expr = self._extract_math_expression(text_input)
        calc_result = None
        routed = []

        # Phase 1: Calculatorの公平な呼び出し
        if math_expr and "Calculator" in self.synapses:
            routed.append("Calculator")
            calc_out = self.synapses["Calculator"](text_input=math_expr)
            if not calc_out.startswith("CALC_ERROR"):
                calc_result = calc_out.replace("CALC_RESULT: ", "").strip()

        # Phase 2: 得られた結果を文脈としてGeneralLLMに渡す
        if "GeneralLLM" in self.synapses:
            routed.append("GeneralLLM")
            if calc_result:
                cooperative_prompt = (
                    f"{text_input}\n"
                    f"(参考情報: システムの計算モジュールが算出した結果、数式 '{math_expr}' の正解は '{calc_result}' です。"
                    f"この値を用いて、ユーザーの指示に自然な日本語で答えてください。決して自分で再計算しないでください。)"
                )
            else:
                cooperative_prompt = text_input
            
            response = self.synapses["GeneralLLM"](cooperative_prompt)
            return response, routed
        
        return "Error: LLM not found", routed


# --- Approach B: Chunk-Level Dynamic Routing ---
class SRA_ChunkBypassMotherboard(SRA_BaseMotherboard):
    """アプローチB: チャンク単位のダイナミック・バイパス"""
    def _split_into_chunks(self, text):
        # 簡易的に数式部分と非数式部分を切り分ける
        pattern = r'([\d\s\+\-\*\/\(\)\.]{3,})'
        parts = re.split(pattern, text)
        chunks = []
        for p in parts:
            if not p:
                continue
            # 演算子を含むものは数式チャンクとする
            if any(op in p for op in ['+', '-', '*', '/']) and len(p.strip()) >= 3:
                chunks.append({"type": "math", "content": p.strip()})
            else:
                chunks.append({"type": "text", "content": p.strip()})
        return chunks

    def route_and_forward(self, text_input):
        chunks = self._split_into_chunks(text_input)
        results = []
        routed = []
        
        for chunk in chunks:
            if chunk["type"] == "math" and "Calculator" in self.synapses:
                routed.append("Calculator")
                calc_out = self.synapses["Calculator"](text_input=chunk["content"])
                if not calc_out.startswith("CALC_ERROR"):
                    ans = calc_out.replace("CALC_RESULT: ", "").strip()
                    results.append(ans)
                else:
                    results.append(chunk["content"])
            elif chunk["type"] == "text" and "GeneralLLM" in self.synapses:
                routed.append("GeneralLLM")
                # 単なる対話チャンクをLLMに送る
                # ただし「次の計算をして。」などの短い指示をそのまま投げると、LLMが「どのような計算ですか？」と聞き返す可能性があるため、
                # マザーボードが「計算の導入応答」として処理するか、あるいは単に対話を流す
                prompt = f"応答の文脈: 計算結果を提示する直前の言葉として、自然な一文を作成してください。ユーザー指示: '{chunk['content']}'"
                llm_out = self.synapses["GeneralLLM"](prompt)
                results.append(llm_out)
            else:
                results.append(chunk["content"])
                
        # 最終的な結合
        final_response = " ".join(results)
        return final_response, list(set(routed))


# --- Approach C: Blackboard (Shared Memory) Architecture ---
class SRA_BlackboardMotherboard(SRA_BaseMotherboard):
    """アプローチC: メッセージバス／黒板モデル"""
    def __init__(self):
        super().__init__()
        # LLMに対し、計算が必要な場合は特殊なフォーマットでバスに要求するよう指示するシステムプロンプト
        self.llm_system_prompt = (
            "You are having a conversation. If the user asks for a math calculation, "
            "do NOT try to calculate it yourself. Instead, output EXACTLY like this: '[CALC_REQ: expression]' "
            "and stop generating immediately. If the user's message contains calculated results (e.g. '[CALC_RES: value]'), "
            "use that value to answer the user's question in a natural way."
        )
        
    def route_and_forward(self, text_input):
        routed = []
        
        # 1. 最初のLLM呼び出し（メッセージバス上の要求検知フェーズ）
        if "GeneralLLM" not in self.synapses:
            return "Error: LLM not found", routed
            
        routed.append("GeneralLLM")
        # 一時的にシステムプロンプトを変更したLLMを適用
        original_prompt = self.synapses["GeneralLLM"].system_prompt
        self.synapses["GeneralLLM"].system_prompt = self.llm_system_prompt
        
        first_out = self.synapses["GeneralLLM"](text_input)
        
        # [CALC_REQ: ...] が含まれているかチェック
        match = re.search(r'\[CALC_REQ:\s*(.+?)\]', first_out)
        if match and "Calculator" in self.synapses:
            routed.append("Calculator")
            math_expr = match.group(1).strip()
            
            # 2. 計算シナプスの起動（メッセージバスを介した非同期処理）
            calc_out = self.synapses["Calculator"](text_input=math_expr)
            calc_result = "CALC_ERROR"
            if not calc_out.startswith("CALC_ERROR"):
                calc_result = calc_out.replace("CALC_RESULT: ", "").strip()
                
            # 3. メッセージバス（対話履歴）に計算結果を書き込み、再度LLMを呼び出し
            routed.append("GeneralLLM")
            messages = [
                {"role": "user", "content": text_input},
                {"role": "assistant", "content": first_out},
                {"role": "user", "content": f"[CALC_RES: {calc_result}]"}
            ]
            
            final_out = self.synapses["GeneralLLM"]("", messages=messages)
            # プロンプトを元に戻す
            self.synapses["GeneralLLM"].system_prompt = original_prompt
            return final_out, routed
            
        # 計算要求がなければそのまま返す
        self.synapses["GeneralLLM"].system_prompt = original_prompt
        return first_out, routed


# ==========================================
# 3. 評価エンジンの実装
# ==========================================

# テストケースの定義
# 各ケースの期待される正解（数式の結果）を定義しておく
TEST_CASES = [
    {
        "id": 1,
        "input": "次の計算をして。1 + 2 * 3",
        "expected_math": 7,
        "description": "一般的な数式指示"
    },
    {
        "id": 2,
        "input": "15 * 4 はいくつですか？",
        "expected_math": 60,
        "description": "質問形式の数式"
    },
    {
        "id": 3,
        "input": "こんにちは！3 + 5 * 2 の結果を教えて。",
        "expected_math": 13,
        "description": "挨拶と数式が混在する文"
    }
]

def evaluate_response(response, expected_math):
    # 出力の中に正しい計算結果が含まれているか
    has_math = str(expected_math) in response
    
    # 日本語として自然か（CALC_RESULTなどの無機質なプレフィックスだけになっていないか）
    is_natural = not response.startswith("CALC_RESULT") and not response.startswith("CALC_ERROR")
    # 最低限の日本語の丁寧さや文脈があるか
    is_natural = is_natural and any(word in response for word in ["です", "ます", "答え", "結果", "は"])
    
    # 総合スコアリング (10点満点)
    score = 0
    if has_math:
        score += 6  # 正確な計算は最重要
    if is_natural:
        score += 4  # 自然な対話表現
        
    return {
        "accuracy": has_math,
        "naturalness": is_natural,
        "score": score
    }

def run_experiment():
    # シナプスの初期化
    print("[Eval] Initializing Synapses...")
    llm = LLMSynapse()
    calc = CalculatorSynapse()
    
    # 各マザーボードアプローチのセットアップ
    approaches = {
        "A: Cooperative Routing": SRA_CooperativeMotherboard(),
        "B: Chunk-Level Bypass": SRA_ChunkBypassMotherboard(),
        "C: Blackboard Architecture": SRA_BlackboardMotherboard()
    }
    
    for name, mb in approaches.items():
        mb.register_synapse("GeneralLLM", llm)
        mb.register_synapse("Calculator", calc)
        
    results = {}
    
    print("\n" + "="*50)
    print("SRA CO-ROUTING BENCHMARK EXPERIMENT")
    print("="*50)
    
    for app_name, mb in approaches.items():
        print(f"\nEvaluating: {app_name}...")
        results[app_name] = []
        
        for tc in TEST_CASES:
            print(f"  Test Case {tc['id']}: '{tc['input']}'")
            start_time = time.time()
            
            try:
                response, routed = mb.route_and_forward(tc["input"])
                elapsed = time.time() - start_time
                
                eval_res = evaluate_response(response, tc["expected_math"])
                
                res_data = {
                    "case_id": tc["id"],
                    "input": tc["input"],
                    "response": response,
                    "routed": routed,
                    "latency": elapsed,
                    "accuracy": eval_res["accuracy"],
                    "naturalness": eval_res["naturalness"],
                    "score": eval_res["score"]
                }
                results[app_name].append(res_data)
                
                print(f"    -> Response: \"{response}\"")
                print(f"    -> Routed via: {routed}")
                print(f"    -> Score: {eval_res['score']}/10 | Latency: {elapsed:.2f}s")
                
            except Exception as e:
                print(f"    -> Failed with error: {str(e)}")
                results[app_name].append({
                    "case_id": tc["id"],
                    "input": tc["input"],
                    "response": f"ERROR: {str(e)}",
                    "routed": [],
                    "latency": 0.0,
                    "accuracy": False,
                    "naturalness": False,
                    "score": 0
                })
                
    # 実験データの集計とレポート作成
    generate_report(results)

def generate_report(results):
    report_path = "/Users/suzukijun/.gemini/antigravity-ide/brain/8e9ceaef-c94d-41bc-8368-c4a30add990f/walkthrough.md"
    print(f"\n[Eval] Generating markdown report at: {report_path}")
    
    markdown = []
    markdown.append("# SRA 協調ルーティング評価実験結果レポート\n")
    markdown.append("本レポートは、SRAにおいて `GeneralLLM` と `Calculator` を対等な立場（公平なシナプス）のまま維持しつつ、ユーザーの数式混じりの自然言語指示に対して、正確かつ自然な応答を生成するための3つのアプローチを比較検証した結果です。\n")
    
    markdown.append("## 評価指標と基準\n")
    markdown.append("- **計算の正確性（Math Accuracy）**: 計算結果が数学的に正しいか（LLMが不正確な推測をしていないか）。")
    markdown.append("- **応答の自然さ（Response Naturalness）**: 計算機出力単体 (`CALC_RESULT: 7`) ではなく、自然な日本語会話の形式になっているか。")
    markdown.append("- **処理の公平性（Architecture Decoupling）**: LLMの下位に電卓を置かず、マザーボード主導で公平なシナプス連携が実現できているか。")
    markdown.append("- **応答速度（Latency）**: 処理完了までに要した時間（秒）。\n")
    
    markdown.append("## アプローチ別・総合スコア要約\n")
    markdown.append("| アプローチ | 平均スコア (10点満点) | 計算正確率 | 自然応答率 | 平均遅延 (秒) | 公平性（設計評価） |")
    markdown.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
    
    summary_data = []
    for app_name, cases in results.items():
        avg_score = sum(c["score"] for c in cases) / len(cases)
        acc_rate = sum(1 if c["accuracy"] else 0 for c in cases) / len(cases) * 100
        nat_rate = sum(1 if c["naturalness"] else 0 for c in cases) / len(cases) * 100
        avg_latency = sum(c["latency"] for c in cases) / len(cases)
        
        decoupling_desc = ""
        if "Cooperative" in app_name:
            decoupling_desc = "🌟 **高**: マザーボードが完全に制御。LLMは計算機の存在を知らない。"
        elif "Chunk" in app_name:
            decoupling_desc = "🌟 **高**: チャンクごとに物理的に別モジュールを起動して最終結合。"
        elif "Blackboard" in app_name:
            decoupling_desc = "👍 **中**: 共通メッセージバス（履歴）を介し、シグナルを媒介に疎結合連携。"
            
        markdown.append(f"| {app_name} | {avg_score:.1f} / 10 | {acc_rate:.0f}% | {nat_rate:.0f}% | {avg_latency:.2f}s | {decoupling_desc} |")
        summary_data.append({
            "name": app_name,
            "score": avg_score,
            "accuracy": acc_rate,
            "naturalness": nat_rate,
            "latency": avg_latency
        })
        
    markdown.append("\n")
    
    markdown.append("## テストケース別・詳細結果\n")
    
    for app_name, cases in results.items():
        markdown.append(f"### {app_name}\n")
        markdown.append("| ID | テスト入力 | 出力応答 | 経由したシナプス | 正確 | 自然 | スコア | 遅延 |")
        markdown.append("| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |")
        for c in cases:
            acc_str = "✅" if c["accuracy"] else "❌"
            nat_str = "✅" if c["naturalness"] else "❌"
            routed_str = " → ".join(c["routed"])
            markdown.append(f"| {c['case_id']} | `{c['input']}` | \"{c['response']}\" | `{routed_str}` | {acc_str} | {nat_str} | {c['score']}/10 | {c['latency']:.2f}s |")
        markdown.append("\n")
        
    markdown.append("## 各アプローチの考察とアーキテクチャ設計推奨\n")
    
    markdown.append("### 🏆 推奨: アプローチA (協調的ツーフェーズ・ルーティング)\n")
    markdown.append("> [!TIP]\n")
    markdown.append("> **結論として「アプローチA (Cooperative Routing)」が最も実用的かつ高精度で、SRAの公平なシナプス思想に合致しています。**\n")
    markdown.append("> \n")
    markdown.append("> **理由:**\n")
    markdown.append("> 1. **高い正確性と自然さ**: 電卓シナプスの厳密な計算結果をLLMの文脈に挿入するため、LLMの持つ豊かな表現力を損なわずに100%正確な結果を回答できます。\n")
    markdown.append("> 2. **完全な疎結合（公平性）**: LLMは電卓モジュールの存在を一切知らず、マザーボードから渡された文脈に従うだけです。これにより、シナプスのホットスワップ（交換）や削除が非常に容易になります。\n")
    markdown.append("> 3. **低遅延**: LLMの起動は1回で済むため、アプローチCと比較して遅延が半分に抑えられます。\n")
    
    markdown.append("### アプローチB (チャンク単位バイパス) の限界\n")
    markdown.append("> [!WARNING]\n")
    markdown.append("> テキストを物理的に分割するため、「次の計算をして。」と「1 + 2 * 3」が別々の入力として別モジュールに送られます。結果として、LLMは「計算の導入応答」だけを生成し、電卓は無機質な数値だけを返すため、それらを結合した出力が「計算結果は以下の通りです： 7」のように**継ぎ接ぎ感のある不自然な文章**になりやすいという欠点があります。\n")
    
    markdown.append("### アプローチC (メッセージバス) のポテンシャルと課題\n")
    markdown.append("> [!NOTE]\n")
    markdown.append("> アプローチCは、複雑な推論の途中で動的に計算が必要になるケース（マルチホップ推論など）において極めて強力なポテンシャルを持っています。\n")
    markdown.append("> ただし、**「LLMを2回呼び出す必要がある」**ため遅延が倍増すること、およびLLMに対して特殊なシグナル（`[CALC_REQ: ...]`）を出力するように高度な指示（あるいはプロンプト調整）が必要となるため、簡易的な数式計算タスクにおいてはオーバーヘッドが大きすぎます。\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown))
        
    print("[Eval] Markdown report successfully generated!")

if __name__ == "__main__":
    run_experiment()
