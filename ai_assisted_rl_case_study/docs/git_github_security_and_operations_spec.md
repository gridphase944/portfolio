# Git・GitHubセキュリティおよび運用仕様

> **Historical snapshot: 2026年8月時点**  
> 本文書は当時のAI-assisted開発運用を示すための公開用snapshotであり、現在のモデル・agent挙動・推奨設定に対する最新ガイドではない。  
> 公開copy：研究開発プロセスの意味を保持し、非公開文書へのリンクと環境固有のpath表記のみ公開向けに整理した。参照する非公開文書は本packageに含めない。

本書は、`daytrading_rl_system`におけるGit・GitHub操作、機密情報、data・artifactの保存境界、remote操作、GitHub Actions、および漏洩疑い時の対応を定める共通仕様である。

本書の存在だけでは、commit、push、merge、history rewrite、force操作、repository設定変更を承認しない。

## 1. 目的・対象・正本

### 1.1 目的

- secret、認証情報、口座情報、個人情報をGit履歴やGitHubへ混入させない。
- Codexが既存dirty stateや他者の変更を誤ってstage、commit、削除しないようにする。
- local変更、commit、remote操作、不可逆操作の権限境界を明確にする。
- raw data、checkpoint、weights、巨大artifactをrepositoryから分離する。
- 漏洩疑い時に、file削除だけで解決扱いせず、credentialと履歴の両方を扱う。

### 1.2 対象と非対象

対象：Git working tree、index、commit、branch、tag、remote、push、pull request、merge、rebase、history rewrite、GitHub repository設定、Actions、Secrets、artifact、secret incident。

非対象：paper／live trading、broker認証方式、OS全体のsecret管理、cloud IAMの詳細。これらは別の正式仕様とユーザー承認を必要とする。

### 1.3 正本

- actorの権限と承認は、現在のユーザー指示とproject instructionを正本とする。
- repository内の作業手順は`AGENTS.md`を正本とする。
- raw dataとGit保存境界はData Inventory（本公開packageには含めない）を参照する。
- handoff bundleの内容と禁止物は[Handoff Bundle Policy](handoff_bundle_policy.md)を正本とする。
- experiment artifactは[ML・RL実験ガバナンス仕様](ml_rl_experiment_governance_spec.md)を参照する。

個別依頼は本書より厳しい制限を追加できるが、明示的なユーザー承認なしに本書を緩和しない。

## 2. 操作権限

Git操作は次のlevelで区別する。

| level | operation | default |
|---|---|---|
| R0 | `status`、`diff`、`log`、`show`、`rev-parse`、`ls-files`、`check-ignore`等のread-only確認 | taskに必要なら可 |
| R1 | 個別依頼で許可されたfileのworking tree編集 | 許可範囲内のみ可 |
| R2 | `git add`、index変更、stash作成・適用 | 明示許可が必要 |
| R3 | commit、amend、branch／tag作成 | 明示許可が必要 |
| R4 | push、pull request作成・更新、merge | 明示許可が必要 |
| R5 | rebase、cherry-pick、reset、restore、clean、branch／tag削除、remote変更 | 個別の明示許可が必要 |
| R6 | force push、history rewrite、repository設定、Actions／Secrets／ruleset変更 | ユーザーの明示承認と専用計画が必要 |

個別依頼で許可されていないlevelへ自動的に進まない。上位levelの許可から別種類の操作を推測しない。

`git reset --hard`、`git clean`、force push、history rewrite等を、作業を簡単にする目的で使用しない。

## 3. 開始時・終了時のGit状態

変更作業では、開始時と終了時に少なくとも次を記録する。

- repository root、HEAD、branch
- staged、tracked modified／deleted、untracked
- worktree、submodule
- 対象fileと既存dirty state
- taskで追加・変更・削除したfile

開始時から存在するdirty stateはtask入力として扱い、無断でrestore、stage、commit、stash、削除しない。

Codexは、自分の変更と開始前からの変更を区別する。対象外fileの状態が変化した場合、原因を確認するまで完了扱いにしない。

## 4. 機密情報と禁止物

private repositoryであることを安全性の根拠にしない。

次をsource、config、test fixture、log、Markdown、commit message、branch名、tag、issue、pull request、Discussion、Wiki、Actions log、artifactへ記録しない。

- secret、token、API key、password、private key、credential
- 認証file、session、cookie、local auth、口座認証情報
- 証券口座情報、個人情報
- secretを復元できる暗号化前の値、URL、header、raw response

exampleやtestには実credentialと区別できる明示的なdummy値だけを使う。

CodexとChatGPTへsecret本文を貼り付けない。必要な確認は、path、種類、存在、mask済みfingerprint等に限定する。

## 5. Data・artifactの保存境界

原則として次をGit管理しない。

- raw data、processed data全量
- `outputs/`内のrun artifact、handoff bundle
- model weights、checkpoint、ReplayBuffer、cache
- TensorBoard event、full prediction、full transition／episode dump
- 巨大log、CSV、JSON、ZIP、Docker image
- local config、認証file、secret

小型で長期参照する正式仕様は`docs/`、versioned small config／splitは`config/`、過去判断のprovenanceは`history/`へ置く。run固有の結果は`outputs/`へ置く。

Git LFSは、上記禁止物をrepositoryへ追加する許可にはならない。

handoff bundleへ含める内容は[Handoff Bundle Policy](handoff_bundle_policy.md)に従う。secret、raw data、weights、checkpointを含めない。

## 6. `.gitignore`・`.dockerignore`

Git対象外fileは`.gitignore`へ、Docker build contextからも除外すべきlocal／secret／巨大fileは`.dockerignore`へ反映する。

ignore pattern追加後は、代表pathについて`git check-ignore`等で意図どおりか確認する。必要なsource、test、仕様、small configを過剰にignoreしない。

`.gitignore`は既にtrackedのfileへ効果を持たない。tracked禁止物を発見した場合、ignore追加だけで解決扱いにせず、index、履歴、remoteへの影響を確認する。

ignoreされたautomationやagent設定は通常の`git status`で見えない可能性があるため、該当taskでは対象pathを明示的に確認する。

## 7. Stage・commit

stageまたはcommitが許可された場合も、対象pathを明示的に選ぶ。既存dirty stateを巻き込む`git add -A`、`git add .`等は、全差分を確認し個別依頼で許可された場合を除き使用しない。

commit前に少なくとも次を確認する。

- staged diffがtask範囲と一致する
- secret、local path、raw data、巨大artifactがない
- 必要なtest、validator、`git diff --check`が完了している
- generated fileやoutputを誤ってstageしていない
- commit messageに機密情報や誤った承認状態がない

commit、amend、squash、署名、tagは、個別依頼で指定された範囲だけ行う。

## 8. Push・PR・merge・branch

remoteへ作用する操作は、repository、remote URL、branch、対象commitを確認した後、明示許可された場合だけ行う。

push前に、local-only commitの範囲、remote差分、secret／large file、test結果を確認する。

force push、branch／tag削除、default branch変更、repository移管、visibility変更を通常作業として行わない。

重要branchでは、利用可能ならbranch protectionまたはrulesetを使用し、force push／削除の禁止、required review、required status checks等を検討する。ただしrepository設定の変更は別途ユーザー承認を必要とする。

pull requestやmergeは、実装healthやreviewの完了を示し得るが、ML・RL candidateの採用またはformal承認を自動的に意味しない。

## 9. GitHub Actions・Secrets

GitHub Actionsを追加・変更する場合は、workflowのtrigger、権限、network、secret、artifact、third-party actionを明示的にreviewする。

- `GITHUB_TOKEN`とworkflow permissionsは必要最小限にする。
- third-party actionはsourceと権限を確認し、原則としてfull-length commit SHAへpinする。
- untrusted codeを実行するjobへsecretを渡さない。
- secretをlog、cache、artifact、job outputへ出さない。
- cloud認証では利用可能なら長期secretよりOIDC等の短期credentialを検討する。
- workflow artifactへraw data、weights、checkpoint、credentialを含めない。

Actions、Secrets、runner、environment、repository設定の変更はR6として扱う。

## 10. Secret scanningとpush protection

GitHubのsecret scanning、push protection、ruleset等がrepository planで利用可能な場合は、有効化を推奨する。

これらは補助防御であり、local確認、ignore、review、権限制御を代替しない。検出対象外のcredentialやcustom secretが存在し得るため、未検出を安全の証明にしない。

push protectionをbypassする場合は、false positiveまたは安全なdummyであることを確認し、理由を記録する。実credentialを「後で修正する」としてbypassしない。

## 11. 漏洩疑い時の対応

secret、credential、個人情報等をcommitまたはpushした可能性がある場合、通常作業を止め、`SECURITY_INCIDENT_REVIEW`として扱う。

1. 追加push、merge、CI、artifact公開を止める。
2. 対象の種類、file、commit、branch、tag、remote、PR／issue／Actions log／artifactへの露出範囲を確認する。
3. secretの場合は、履歴修正より先に失効・rotationを行う。Codexはcredential値を取得・操作せず、ユーザーへ必要対応を報告する。
4. local file削除だけで安全になったと判断しない。
5. 履歴rewriteが必要かをPM判断し、実施時はclone、fork、branch、tag、PR、automation、commit hash変更の影響を含む専用計画を作る。
6. cleanup後に再scanし、再混入防止のignore、validator、push protection等を追加する。
7. incident、rotation、history cleanup、残存riskを小型receiptへ記録する。secret本文は記録しない。

history rewriteとforce pushはR6であり、incidentであってもCodexが独断で実行しない。

## 12. Validation・証拠・導入

適用範囲に応じて次を確認する。

- `git status`、staged／unstaged diff、対象外dirty state
- `git ls-files`、`git check-ignore`
- secret pattern／credential file／large artifactの混入
- JSON、TOML、YAML、Markdown、workflowのparse
- `.gitignore`／`.dockerignore`の代表match
- Actions permissions、third-party action pin、artifact内容
- commit、push、merge、history rewriteの実行有無
- 開始時／終了時Git状態

Codexの最終報告は、実行したGit操作、変更file、test／validator、commit、push、remote操作、残るriskを区別して示す。

本仕様は承認後の新規Git・GitHub作業へ適用する。既存historyを本仕様に合わせて自動rewriteせず、過去の機密情報が疑われる場合だけ別途incident reviewを行う。
