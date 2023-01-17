# TODO Works
## Motion Matching
- [x] 実装したマッチングアルゴリズムが実時間上で動くことを確認(複数回) - 現状1回あたり0.33s 
- [ ] FAISSを用いたGPUマッチングの実装 - faiss-gpuがAnacondaからインストールできない？
- [ ] [`anim/pose.py`](anim/pose.py)に`Pose`クラスを実装。関節位置などを保存する入れ子とする。
- [ ] 予め与えられたカーブからのアニメーション作成(オフライン、事前にsimulation boneの位置と向きを配列(T, pos + dir)に保存し、それに沿うようなアニメーションを作成する)
- [ ] キーボード入力に対応
- [ ] シミュレーションの作成(キーボードに対応させてsimulation boneを移動、回転)
- [ ] 将来の軌道カーブの作成(キーボード入力から作成)
- [ ] グラフィックスインターフェイスの用意(PyOpenGL or Taichi?)
- [ ] InertializationやIKを無視した、単なるポーズ再生の切り替えで動くことを確認(リアルタイム)
- [ ] ダンピングアルゴリズムの実装([`anim/blend.py`](anim/blend.py))
- [ ] FootIKを[`Animation`](anim/animation.py)に取り入れる
- [ ] FootIKや Inertializationによる自然なアニメーションの作成(リアルタイム)
- [ ] AMASSデータも使えるようにする

## その他
- [ ] Motion Blendの完全な実装
- [ ] Motion Graphsの作成(MMの機能を一部コピー)
- [ ] FullbodyIKを[`Animation`](anim/animation.py)に取り入れる(Jacobian and FABRIK)。
- [ ] Learned Motion Matchingの追実装
- [ ] Crowd animationの方策を検討
- [ ] LBSの実装
