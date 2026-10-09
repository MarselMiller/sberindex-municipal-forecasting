# Публикация существующего репозитория

Автор выбрал `MarselMiller/sberindex-municipal-forecasting` с сохранением всей
Git history. Команды ниже выполняет владелец **после review кандидата**.
Commit/push/merge, visibility и deployment при подготовке не выполнялись.
В PowerShell выполняйте команды по очереди; при ошибке остановитесь.

## 1. Commit и push release-ветки

Из корня проекта. Кандидат подготовлен в index явно перечисленными файлами;
не используйте `git add -A` для добавления непроверенных материалов.

```powershell
git switch release/github-pages
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached
git ls-files --error-unmatch -- reports/results/e08b/feature_coverage.csv .github/workflows/pages.yml requirements-publication.txt
git commit -m "Prepare reviewed public repository and manual GitHub Pages publication"
git push --set-upstream origin release/github-pages
```

Это сохраняет существующие commits. Лицензия собственного кода не назначается;
MIT не добавлена. Атрибуция и CC BY-SA 4.0 относятся к примеру МО 21 и его
адаптированному материалу; [точный scope](../reports/final/figure_data/README.md).

## 2. Merge в main без переписывания истории

Из корня исходного проекта, после успешного review. Если `origin/main` изменился
после проверенного `4b0c1b6`, сначала согласуйте и проверьте получившийся merge.
Не используйте squash, rebase или force push.

```powershell
git switch main
git pull --ff-only origin main
git merge --no-ff release/github-pages -m "Merge reviewed GitHub Pages release"
git diff --check
git push origin main
```

Альтернатива UI: pull request `release/github-pages` → `main`, обычный
**Create a merge commit** после review. Выполните один из двух способов merge.

## 3. GitHub-only clean-clone verification

Из корня исходного проекта создаётся новая папка проверки под ignored `outputs/`.
Из исходной рабочей папки ничего не копируется. Clone получает также remote
branches для history audit. Для проверки release перед merge замените только
`$publicationBranch` на `release/github-pages`; для окончательной проверки нужен `main`.

```powershell
$publicationBranch = 'main'
$verificationDir = Join-Path (Get-Location) ('outputs/github-only-check-' + [guid]::NewGuid().ToString('N'))
git clone --branch $publicationBranch https://github.com/MarselMiller/sberindex-municipal-forecasting.git $verificationDir
Set-Location -LiteralPath $verificationDir
git rev-parse HEAD
git status --short
py -3.12 -m venv .venv-publication
$publicationPython = Join-Path (Get-Location) '.venv-publication/Scripts/python.exe'
& $publicationPython -m pip install -r requirements-publication.txt
& $publicationPython -m pip check
& $publicationPython -X utf8 scripts/build_final_summary.py --editorial-only --check
& $publicationPython -X utf8 scripts/build_project_report.py --check
& $publicationPython -X utf8 scripts/build_publication_site.py --require-tracked-inputs
& $publicationPython -X utf8 scripts/build_publication_site.py --require-tracked-inputs --check
& $publicationPython -m pytest -q -p no:cacheprovider --basetemp=outputs/publication_pytest tests/test_publication_site.py tests/test_public_editorial.py tests/test_public_repository_audit.py tests/test_final_summary.py
& $publicationPython -X utf8 scripts/audit_public_repository.py --strict
& $publicationPython -X utf8 scripts/verify_publication_browser.py --project-prefix
git diff --check
git status --short
```

Ожидается: все команды успешны, 57 no-fit tests, package/link/browser PASS,
security `PASS_WITH_REVIEW` без BLOCKER и только с принятыми владельцем замечаниями,
финальный `git status --short` пуст. Browser QA требует установленный Edge в Windows;
сам сайт и сборка не требуют Edge. Private inputs/outputs и ML packages не нужны.
Не запускайте полный pytest или экспериментальные runners в этой проверке.

## 4. Короткая инструкция visibility и Pages

1. В существующем репозитории откройте **Settings → General → Danger Zone →
   Change repository visibility → Public** и подтвердите выбранный repository.
   [Официальная инструкция GitHub](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility).
2. Откройте **Settings → Pages → Build and deployment → Source → GitHub Actions**.
   Workflow собирает `dist/submission-site/`; выбирать `/docs` не нужно.
   [Custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
3. В **Actions → GitHub Pages → Run workflow** выберите **main**. Первый запуск
   с `deploy=false` проверяет сборку и загружает artifact, не публикуя сайт.
   После успешной проверки отдельно запустите с **deploy=true**.
   [Ручной запуск workflow](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
4. Дождитесь успешных `build` и `deploy`, затем откройте
   [ожидаемый URL](https://marselmiller.github.io/sberindex-municipal-forecasting/)
   без авторизации: проверьте графики, glossary search, PDF, методологию и E08/Persistence/
   Robustness приложения на настольной и мобильной ширине. Проверьте Network/Console
   на 404 и ошибки, а адрес результата сверяйте с output `deploy-pages`.

Сайт до успешного deployment не объявляется опубликованным. README сохраняет
формулировку «планируемый адрес» до фактической проверки URL. Параметр `deploy`
по умолчанию false, workflow не имеет push/PR/schedule триггеров.

Технический состав и границы проверок — [README_PUBLICATION.md](README_PUBLICATION.md).
Первичные условия и решение автора — [PUBLICATION_RIGHTS_REVIEW.md](../reports/final/PUBLICATION_RIGHTS_REVIEW.md).
