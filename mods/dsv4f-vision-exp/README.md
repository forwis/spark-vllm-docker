# DeepSeek V4 Flash Vision-Exp on JASL DSpark v2

This mod ports only the DeepSeek V4 Vision Python runtime delta onto the locally
built `vllm-node-dsv4f:v2` image. It requires JASL vLLM commit
`9ad62027bc84ca0ccbcc40853179312de770220c`; unknown layouts fail before vLLM
starts.

The mod registers `DeepseekV4VForConditionalGeneration`, installs the Vision
tower and processor, enables image sentinels and `bias_vl` routing, and maps
the Vision gate bias into DSpark draft layers. It does not replace CUDA kernels
or install a different vLLM tree.

Use the qualified recipe:

```bash
./run-recipe.sh deepseek-v4-flash-vision-exp --setup
```

The recipe is two-node only. Development tests do not prove GPU correctness;
acceptance requires text, image, and DSpark smoke tests on the Spark pair.
Remove the recipe's `mods/dsv4f-vision-exp` entry to roll back the runtime
patch, and use the unmodified local v2 image for text-only serving.

The complete upstream provenance and license record is in
[`UPSTREAM.md`](UPSTREAM.md).

The recipe deliberately uses `fp8_ds_mla`: the reference profile's padded
`nvfp4_ds_mla` name routes the same FP8 layout and supplies no 4-bit capacity
gain.
