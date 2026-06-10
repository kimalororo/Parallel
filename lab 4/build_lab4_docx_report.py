from __future__ import annotations

import csv
import datetime as dt
import math
import re
from pathlib import Path

import win32com.client
from PIL import Image


OUTPUT = Path("Отчет_лаб4.docx").resolve()
FALLBACK_OUTPUT = Path("Отчет_лаб4_исправленный.docx").resolve()
PDF_OUTPUT = Path("outputs/docx_render_word/report_lab4.pdf").resolve()

WD_REPLACE_ALL = 2
WD_PAGE_BREAK = 7
WD_FORMAT_XML_DOCUMENT = 12
WD_EXPORT_FORMAT_PDF = 17
WD_STATISTIC_PAGES = 2
WD_LINE_STYLE_SINGLE = 1
WD_AUTO_FIT_WINDOW = 2


def find_template() -> Path:
    candidates = [
        p
        for p in Path(".").glob("*.docx")
        if not p.name.startswith("~$")
        and not p.name.startswith("Task 4")
        and not p.name.startswith("Отчет_лаб4")
    ]
    if not candidates:
        raise FileNotFoundError("Не найден шаблон отчета в формате .docx")
    return max(candidates, key=lambda p: p.stat().st_size).resolve()


def style(doc, *names: str):
    for name in names:
        try:
            return doc.Styles(name)
        except Exception:
            continue
    raise KeyError(f"Не найден стиль: {names}")


def replace_all(doc, old: str, new: str) -> None:
    rng = doc.Content
    find = rng.Find
    find.ClearFormatting()
    find.Replacement.ClearFormatting()
    find.Execute(FindText=old, ReplaceWith=new, Replace=WD_REPLACE_ALL)


def delete_template_body(doc) -> None:
    rng = doc.Content
    find = rng.Find
    find.ClearFormatting()
    find.Text = "Задание"
    if find.Execute():
        doc.Range(rng.Start, doc.Content.End - 1).Delete()


def set_paragraph(selection, doc, text: str, style_name: str = "Основной текст1") -> None:
    selection.Style = style(doc, style_name)
    selection.TypeText(text)
    selection.TypeParagraph()


def set_heading(selection, doc, text: str, level: int) -> None:
    if level == 1:
        set_paragraph(selection, doc, text, "*Заголовок отчёта")
    elif level == 2:
        set_paragraph(selection, doc, text, "*Подраздел основной части")
    else:
        set_paragraph(selection, doc, text, "*Пункт основной части")


def set_image_caption(selection, doc, text: str) -> None:
    set_paragraph(selection, doc, text, "Подпись к рисунку")


def set_table_caption(selection, doc, text: str) -> None:
    set_paragraph(selection, doc, text, "Подпись к таблице")


def set_code_block(selection, doc, title: str, lines: list[str]) -> None:
    set_table_caption(selection, doc, title)
    for line in lines:
        selection.Style = style(doc, "Листинг кода")
        selection.TypeText(line.replace("\t", "    ") if line else " ")
        selection.TypeParagraph()


def insert_page_break(selection) -> None:
    selection.InsertBreak(WD_PAGE_BREAK)


def insert_picture(selection, path: Path, max_width_pt: float = 430.0) -> None:
    shape = selection.InlineShapes.AddPicture(
        FileName=str(path.resolve()),
        LinkToFile=False,
        SaveWithDocument=True,
    )
    try:
        shape.LockAspectRatio = True
        if float(shape.Width) > max_width_pt:
            shape.Width = max_width_pt
    except Exception:
        pass
    selection.TypeParagraph()


def insert_table(selection, doc, rows: list[list[str]]) -> None:
    row_count = len(rows)
    col_count = max(len(row) for row in rows)
    tbl = doc.Tables.Add(selection.Range, row_count, col_count)
    tbl.Borders.Enable = True
    tbl.Range.Style = style(doc, "Основной текст1")
    for row_index, row in enumerate(rows, start=1):
        for col_index in range(1, col_count + 1):
            value = row[col_index - 1] if col_index <= len(row) else ""
            cell_range = tbl.Cell(row_index, col_index).Range
            cell_range.Text = str(value)
    tbl.Rows(1).Range.Bold = True
    try:
        tbl.AutoFitBehavior(WD_AUTO_FIT_WINDOW)
    except Exception:
        pass
    tbl.Range.InsertParagraphAfter()
    selection.SetRange(tbl.Range.End + 1, tbl.Range.End + 1)


def extract_animation_frames() -> list[Path]:
    gif = Path("outputs/agent_path_success.gif")
    if not gif.exists():
        gif = Path("outputs/agent_path_sample.gif")
    out_dir = Path("outputs/docx_build")
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(gif)
    indexes = sorted({0, max(0, image.n_frames // 4), max(0, image.n_frames // 2), max(0, image.n_frames - 1)})
    paths: list[Path] = []
    for order, index in enumerate(indexes, start=1):
        image.seek(index)
        frame = image.convert("RGB")
        path = out_dir / f"animation_frame_{order}.png"
        frame.save(path)
        paths.append(path)
    return paths


def read_summary() -> list[dict[str, str]]:
    with Path("outputs/summary_results.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_float(value: str | float, digits: int = 2) -> str:
    number = float(value)
    if math.isnan(number):
        return "нет успешных эпизодов"
    return f"{number:.{digits}f}"


def best_strategy_text(summary: list[dict[str, str]]) -> str:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in summary:
        grouped.setdefault(row["strategy_title"], []).append(row)

    best_title = ""
    best_success = -1.0
    best_steps = 0.0
    for title, rows in grouped.items():
        success = sum(float(row["success_probability"]) for row in rows) / len(rows)
        steps_values = [float(row["avg_success_steps"]) for row in rows if row["avg_success_steps"] != "nan"]
        steps = sum(steps_values) / len(steps_values)
        if success > best_success:
            best_title, best_success, best_steps = title, success, steps

    return (
        f"Наилучший результат показала стратегия «{best_title}»: средняя вероятность успеха "
        f"по всем скоростям генерации составила {best_success:.2f}, "
        f"среднее число шагов успешной доставки — {best_steps:.1f}."
    )


def source_snippet(start_marker: str, end_marker: str | None = None, max_lines: int = 24) -> list[str]:
    lines = Path("lab4_multiprocessing_simulation.py").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if start_marker in line)
    if end_marker:
        end = next(i for i in range(start + 1, len(lines)) if end_marker in lines[i])
    else:
        end = min(len(lines), start + max_lines)
    snippet = lines[start:end]
    return snippet[:max_lines]


def short_list_item(selection, doc, text: str) -> None:
    selection.Style = style(doc, "Элемент списка")
    selection.TypeText(text)
    selection.TypeParagraph()


def build_report() -> tuple[Path, int]:
    template = find_template()
    frames = extract_animation_frames()
    summary = read_summary()

    output = OUTPUT
    if output.exists():
        try:
            output.unlink()
        except PermissionError:
            output = FALLBACK_OUTPUT
            if output.exists():
                output.unlink()
    PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if PDF_OUTPUT.exists():
        PDF_OUTPUT.unlink()

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = None
    try:
        doc = word.Documents.Open(str(template), ReadOnly=False, AddToRecentFiles=False)
        replace_all(doc, "Работа с данными с помощью Apache Spark", "Реализация многоагентной среды с помощью многопроцессных вычислений")
        replace_all(doc, "Лабораторная работа № 1", "Лабораторная работа № 4")
        replace_all(doc, "Вариант - 1", "Вариант - не задан")
        replace_all(doc, "Машинное обучение", "Параллельные вычисления")
        replace_all(doc, "21.05.2026", dt.date.today().strftime("%d.%m.%Y"))

        delete_template_body(doc)
        selection = word.Selection
        selection.EndKey(Unit=6)

        set_paragraph(selection, doc, "Задание", "Заголовок отчёта")
        set_paragraph(selection, doc, "Цель работы — получить навыки создания многоагентной среды с неопределенностью с применением многопроцессных вычислений.")
        set_paragraph(selection, doc, "В лабораторной работе требуется смоделировать движение агента-доставщика и других транспортных средств по городской карте. Движение доставщика выполняется отдельным процессом, движение транспортного потока — вторым процессом. Нужно исследовать, как скорость генерации транспорта и стратегия движения агента влияют на вероятность успешной доставки и среднее число шагов.")
        set_paragraph(selection, doc, "По требованиям задания отчет должен содержать карту области, графики зависимости результата от скорости генерации транспорта и стратегии, визуализацию движения, поясненный программный код и выводы.")
        insert_page_break(selection)

        set_heading(selection, doc, "Основная часть", 1)
        set_heading(selection, doc, "Карта многоагентной среды", 2)
        set_paragraph(selection, doc, "Карта хранится в файле citymap.txt и имеет размер 31 x 31 клетка. Символы карты интерпретируются следующим образом: 0 — дорога, 1 — препятствие, 2 — стартовые позиции доставщика, 3 — точки генерации транспорта, 4 — точка доставки.")
        insert_picture(selection, Path("outputs/map.png"), 390)
        set_image_caption(selection, doc, "Рисунок 1 — Карта области движения агента и транспортных средств")
        set_paragraph(selection, doc, "На карте заданы три стартовые позиции агента: (1, 1), (1, 29), (29, 1), шесть точек появления транспорта: (1, 13), (5, 5), (13, 1), (13, 29), (25, 25), (29, 13), а также цель доставки (29, 29).")

        set_heading(selection, doc, "Правила движения агентов", 2)
        set_paragraph(selection, doc, "На каждой итерации доставщик и каждый бот могут выполнить одно из пяти действий: вверх, вниз, влево, вправо или остановка. Движение допускается только по клеткам 0, 2, 3 и 4. Если транспортное средство попадает в тупик, оно разворачивается. На перекрестке бот с вероятностью 10% останавливается, а на следующей итерации едет в обратную сторону; остальные варианты движения выбираются равномерно из доступных направлений, кроме обратного.")
        set_paragraph(selection, doc, "При появлении из клетки 3 бот получает случайное число шагов жизни от 15 до 150. После исчерпания этого числа шагов бот удаляется с карты, что моделирует парковку или заезд в гараж.")
        set_paragraph(selection, doc, "Столкновение считается аварией. Если сталкиваются боты, они удаляются из среды. Если в столкновении участвует доставщик, эпизод считается неуспешным. Дополнительно учитывается обмен клетками за одну итерацию, когда два участника как бы проезжают друг сквозь друга.")

        set_heading(selection, doc, "Стратегии движения доставщика", 2)
        set_table_caption(selection, doc, "Таблица 1 — Проверенные стратегии прохождения развилок")
        insert_table(selection, doc, [
            ["Стратегия", "Правило на развилке"],
            ["Правый поворот", "Вес остановки+разворота — 5, правого поворота — 40, прямого движения — 30, левого поворота — 25."],
            ["Прямое движение", "Вес остановки+разворота — 5, прямого движения — 55, правого и левого поворота — по 20."],
            ["Манхеттенское расстояние", "Направления, уменьшающие манхэттенское расстояние до цели, получают повышенный вес. На развилке: остановка+разворот — 5, выгодное направление — 70, остальные направления — 15."],
        ])
        set_paragraph(selection, doc, "Стратегия «Манхеттенское расстояние» не строит полный маршрут, а принимает локальное вероятностное решение: для каждой доступной соседней клетки считается расстояние до цели, и направления, которые это расстояние уменьшают, выбираются чаще. Благодаря случайности агент не становится полностью детерминированным и может обходить некоторые неблагоприятные ситуации, но гарантия успеха отсутствует.")

        set_heading(selection, doc, "Многопроцессная организация программы", 2)
        set_paragraph(selection, doc, "Симуляция построена на модуле multiprocessing. Основной процесс управляет эпизодом и синхронизирует шаги, но само движение доставщика и транспортного потока вычисляется в отдельных процессах. Обмен выполняется через Pipe-соединения: главный процесс отправляет состояние и получает обновленное положение.")
        set_code_block(selection, doc, "Листинг 1 — Создание отдельных процессов агента и транспортного потока", source_snippet("class ProcessController:", "    def reset", 22))
        set_paragraph(selection, doc, "В этом фрагменте видно разделение вычислений: agent_worker отвечает за решение доставщика, traffic_worker — за обновление списка ботов. Метод reset нужен для повторяемых экспериментов с разными seed, а основной процесс получает результаты обоих рабочих процессов на каждой итерации.")

        set_heading(selection, doc, "Выбор направления агентом", 2)
        set_paragraph(selection, doc, "Для стратегии «Манхеттенское расстояние» используется одноименная метрика. Она хорошо подходит для клеточной карты с движением по четырем направлениям, потому что измеряет минимальное число вертикальных и горизонтальных шагов без учета диагоналей.")
        target_code = source_snippet("def manhattan", "def load_city_map", 12) + ["..."] + source_snippet('elif strategy == "target_biased":', "    else:", 12)
        set_code_block(selection, doc, "Листинг 2 — Расчет расстояния и выбор направления, приближающего к цели", target_code)
        set_paragraph(selection, doc, "Если возможный шаг уменьшает расстояние до цели, он получает вес 70; остальные допустимые направления получают вес 15, а остановка с последующим разворотом — вес 5. После этого функция weighted_choice превращает веса в вероятности. Например, если на развилке есть одно выгодное и два невыгодных направления, вероятность выгодного шага будет 70 / (70 + 15 + 15 + 5), то есть примерно 66,7%.")

        set_heading(selection, doc, "Движение транспорта и аварии", 2)
        set_code_block(selection, doc, "Листинг 3 — Обновление транспортного потока и генерация новых ботов", source_snippet("def traffic_step", "def agent_worker", 30))
        set_paragraph(selection, doc, "Функция traffic_step сначала двигает уже существующие транспортные средства, затем удаляет ботов, попавших в аварии, и только после этого с заданной вероятностью создает новых ботов в точках генерации. Такой порядок предотвращает появление транспорта в уже занятой клетке.")

        set_heading(selection, doc, "Проведение эксперимента", 2)
        set_paragraph(selection, doc, "Эксперименты проводились для четырех вероятностей генерации транспорта из каждой точки за итерацию: 0,005; 0,010; 0,020; 0,040. Для каждой пары «стратегия — вероятность генерации» выполнено 45 эпизодов: по 15 эпизодов из каждой из трех стартовых точек. Максимальная длина эпизода составляла 350 шагов.")
        insert_picture(selection, Path("outputs/success_probability_by_strategy.png"), 430)
        set_image_caption(selection, doc, "Рисунок 2 — Вероятность успешной доставки при разных стратегиях и скоростях генерации транспорта")
        insert_picture(selection, Path("outputs/avg_steps_by_strategy.png"), 430)
        set_image_caption(selection, doc, "Рисунок 3 — Среднее число шагов успешной доставки при разных стратегиях и скоростях генерации транспорта")

        result_rows = [["Стратегия", "p генерации", "Эпизодов", "Вероятность успеха", "Средние шаги"]]
        for row in summary:
            result_rows.append([
                row["strategy_title"],
                f"{float(row['spawn_probability']):.3f}",
                row["episodes"],
                format_float(row["success_probability"], 2),
                format_float(row["avg_success_steps"], 1),
            ])
        set_table_caption(selection, doc, "Таблица 2 — Агрегированные результаты экспериментов")
        insert_table(selection, doc, result_rows)
        set_paragraph(selection, doc, "Рост вероятности генерации транспорта увеличивает плотность движения на карте. Поэтому возрастает число аварий с участием агента, а вероятность успешной доставки снижается. Среднее число шагов успешных доставок не обязано меняться строго монотонно: при высокой плотности часть длинных эпизодов завершается аварией или тайм-аутом и не попадает в среднее по успешным доставкам.")

        set_heading(selection, doc, "Визуализация движения", 2)
        set_paragraph(selection, doc, "Для успешного демонстрационного эпизода была создана GIF-анимация outputs/agent_path_success.gif. Так как формат Word обычно показывает только один кадр GIF, в отчет включен набор ключевых кадров эпизода. Красный квадрат обозначает доставщика, фиолетовые круги — транспортные средства.")
        for index, frame in enumerate(frames, start=1):
            insert_picture(selection, frame, 340)
            set_image_caption(selection, doc, f"Рисунок {3 + index} — Кадр {index} успешного демонстрационного эпизода")

        set_heading(selection, doc, "Файлы программы и результатов", 2)
        for item in [
            "citymap.txt — текстовая карта среды;",
            "lab4_multiprocessing_simulation.py — программа симуляции и построения результатов;",
            "outputs/episode_results.csv — результаты отдельных эпизодов;",
            "outputs/summary_results.csv — агрегированная статистика;",
            "outputs/agent_path_success.gif — полная анимация успешной доставки.",
        ]:
            short_list_item(selection, doc, item)

        insert_page_break(selection)
        set_heading(selection, doc, "Заключение", 1)
        set_paragraph(selection, doc, best_strategy_text(summary))
        set_paragraph(selection, doc, "Поставленная задача выполнена: сформирована карта городской среды, реализованы процессы для движения доставщика и транспортного потока, добавлена случайная генерация ботов, обработка столкновений, три стратегии движения агента и экспериментальное сравнение результатов.")
        set_paragraph(selection, doc, "Наиболее устойчивой оказалась стратегия «Манхеттенское расстояние», потому что агент чаще выбирает направления, сокращающие расстояние до точки доставки. Однако эта стратегия остается вероятностной и локальной: она не строит глобальный оптимальный маршрут и не предсказывает будущие положения транспорта, поэтому аварии и неуспешные эпизоды сохраняются.")
        set_paragraph(selection, doc, "Увеличение скорости генерации транспорта снижает вероятность доставки, так как на дорогах появляется больше конфликтных ситуаций. Это подтверждает влияние неопределенности среды на поведение многоагентной системы и показывает практическую пользу многократного статистического эксперимента вместо оценки по одному запуску.")

        pages = doc.ComputeStatistics(WD_STATISTIC_PAGES)
        doc.SaveAs2(str(output), FileFormat=WD_FORMAT_XML_DOCUMENT)
        doc.ExportAsFixedFormat(str(PDF_OUTPUT), WD_EXPORT_FORMAT_PDF)
        return output, int(pages)
    finally:
        if doc is not None:
            doc.Close(SaveChanges=False)
        word.Quit()


def main() -> None:
    output, pages = build_report()
    print(f"Created {output.name}; pages={pages}")


if __name__ == "__main__":
    main()
