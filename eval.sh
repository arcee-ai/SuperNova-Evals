# Install required package
pip install antlr4-python3-runtime==4.11 immutabledict langdetect

python -c "import nltk; nltk.download('punkt')"

MODEL_PATHS=(
  arcee-train/Arcee-SuperNova-v1
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_NAME=$(basename "$MODEL_PATH")
  MODEL_DIR="./results/$MODEL_NAME"
  mkdir -p "$MODEL_DIR"
 
  MODEL_ARGS="parallelize=True,trust_remote_code=True,pretrained=$MODEL_PATH,dtype=bfloat16"
  
  BASE_COMMAND="lm_eval --model hf --model_args $MODEL_ARGS --batch_size 4 --fewshot_as_multiturn --apply_chat_template" # batch size of 2 is sometimes used.

    # IFEval
  $BASE_COMMAND --tasks leaderboard_ifeval --fewshot_as_multiturn --output_path "$MODEL_DIR/ifeval"
  
  # BBH (Big-Bench Hard)
  $BASE_COMMAND --tasks leaderboard_bbh --num_fewshot 3 --fewshot_as_multiturn --output_path "$MODEL_DIR/bbh"

  # MMLU-Pro
  $BASE_COMMAND --tasks leaderboard_mmlu_pro --num_fewshot 5 --fewshot_as_multiturn --output_path "$MODEL_DIR/mmlu_pro"

  # Math Level-5
  $BASE_COMMAND --tasks leaderboard_math_hard --num_fewshot 4 --fewshot_as_multiturn --output_path "$MODEL_DIR/math_hard"

  # TruthfulQA
  $BASE_COMMAND --tasks truthfulqa_mc2 --fewshot_as_multiturn --output_path "$MODEL_DIR/truthfulqa"

  # GSM8K
  $BASE_COMMAND --tasks gsm8k --num_fewshot 5 --fewshot_as_multiturn --output_path "$MODEL_DIR/gsm8k"

done
