$ErrorActionPreference = "Stop"

# ============================================================
# 修改成你的 Windows 项目路径
# ============================================================
$ProjectDir = "E:\myproject\EventVggt"

$TodoFile = Join-Path $ProjectDir "toDo.md"
$LogDir = Join-Path $ProjectDir "codex_scheduled_run"

$FinalMessageFile = Join-Path $LogDir "final_message.md"
$RunLogFile = Join-Path $LogDir "codex_run.log"
$GitBeforeFile = Join-Path $LogDir "git_status_before.txt"
$GitAfterFile = Join-Path $LogDir "git_status_after.txt"
$DiffAfterFile = Join-Path $LogDir "diff_after.patch"

# Windows PowerShell 5.1 向外部程序传递中文时使用 UTF-8
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

"============================================================" |
    Out-File $RunLogFile -Encoding utf8

"Codex scheduled run started: $(Get-Date)" |
    Out-File $RunLogFile -Encoding utf8 -Append

"Project: $ProjectDir" |
    Out-File $RunLogFile -Encoding utf8 -Append

"============================================================" |
    Out-File $RunLogFile -Encoding utf8 -Append

# ============================================================
# 基础检查
# ============================================================

if (-not (Test-Path $ProjectDir -PathType Container)) {
    throw "项目目录不存在：$ProjectDir"
}

if (-not (Test-Path $TodoFile -PathType Leaf)) {
    throw "没有找到任务文件：$TodoFile"
}

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "当前 PowerShell 找不到 codex 命令。请先运行 codex --version 检查。"
}

Push-Location $ProjectDir

try {
    # 记录执行前状态，不修改现有代码
    git status --short 2>&1 |
        Out-File $GitBeforeFile -Encoding utf8

    $TodoContent = Get-Content $TodoFile -Raw -Encoding UTF8

    $Prompt = @"
请完整阅读下面给出的 toDo.md，并直接在当前代码仓库中完成其中的全部任务。

执行要求：

1. 先检查当前项目结构和相关实现，再开始修改。
2. 必须真正修改代码，不要只给出修改建议或示例代码。
3. 按照 toDo.md 中的任务顺序逐项完成。
4. 不要撤销、覆盖或丢弃用户已有的未提交修改。
5. 不要修改与任务无关的文件。
6. 每项任务完成后检查调用位置、张量尺寸、配置参数和兼容性。
7. 修改完成后运行适当的语法检查、静态检查或轻量测试。
8. 不要启动耗时很长的正式训练，除非 toDo.md 明确要求。
9. 检查失败时，分析原因并在合理范围内继续修复。
10. 最后总结：
    - 修改了哪些文件；
    - 每个任务具体如何完成；
    - 执行了哪些检查；
    - 还有哪些未解决的问题或风险。

以下是 toDo.md 的完整内容：

==================== toDo.md 开始 ====================

$TodoContent

==================== toDo.md 结束 ====================
"@

    "Starting Codex at $(Get-Date)" |
        Out-File $RunLogFile -Encoding utf8 -Append

    # “-”表示从标准输入读取完整提示词
    $Prompt |
        & codex exec `
            -C $ProjectDir `
            --sandbox workspace-write `
            --output-last-message $FinalMessageFile `
            - 2>&1 |
        Out-File $RunLogFile -Encoding utf8 -Append

    $CodexExitCode = $LASTEXITCODE

    git status --short 2>&1 |
        Out-File $GitAfterFile -Encoding utf8

    git diff 2>&1 |
        Out-File $DiffAfterFile -Encoding utf8

    "Codex finished at $(Get-Date)" |
        Out-File $RunLogFile -Encoding utf8 -Append

    "Exit code: $CodexExitCode" |
        Out-File $RunLogFile -Encoding utf8 -Append

    exit $CodexExitCode
}
catch {
    "ERROR at $(Get-Date)" |
        Out-File $RunLogFile -Encoding utf8 -Append

    $_ | Out-String |
        Out-File $RunLogFile -Encoding utf8 -Append

    throw
}
finally {
    Pop-Location
}