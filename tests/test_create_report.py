"""Report settings, people resolution and template rendering (no TeX needed)."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import create_report as cr  # noqa: E402

# Synthetic people only.
PEOPLE = {"people": [
    {"code": "QMP1", "name": "Erika Mustermann", "title": "Dr., MSc", "aliases": ["Riki"],
     "emails": ["erika@example.org", "e.m@example.com"]},
    {"code": "QMP2", "name": "Muster", "emails": []},
    {"code": "Student1", "name": "Max Beispiel", "emails": ["max@example.org"]},
]}


class SettingsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.people = self.tmp / "people.json"
        self.people.write_text(json.dumps(PEOPLE), encoding="utf-8")
        self.config = self.tmp / "report_config.json"
        self.local_config = self.tmp / "report_config.local.json"
        self.config.write_text(json.dumps({"authors": ["QMP2"], "approvers": ["QMP2"], "contact": "Lead1",
                                           "public_contact_url": "https://example.org/issues"}), encoding="utf-8")
        self.local_config.write_text(json.dumps({"people_file": str(self.people), "lab_url": "https://example.org/lab#x"}),
                              encoding="utf-8")
        self.old = cr.REPORT_CONFIG_PATH, cr.REPORT_LOCAL_CONFIG_PATH
        cr.REPORT_CONFIG_PATH, cr.REPORT_LOCAL_CONFIG_PATH = self.config, self.local_config

    def tearDown(self):
        cr.REPORT_CONFIG_PATH, cr.REPORT_LOCAL_CONFIG_PATH = self.old
        shutil.rmtree(self.tmp)

    def test_precedence(self):
        internal = cr.load_settings(public=False)
        self.assertEqual(internal["lab_url"], "https://example.org/lab#x")
        public = cr.load_settings(public=True)
        self.assertIsNone(public["lab_url"])
        self.assertIsNone(public["people_file"])
        cli = cr.load_settings(public=False, cli_authors="QMP1, Student1")
        self.assertEqual(cli["authors"], ["QMP1", "Student1"])

    def test_names_are_rejected_as_codes(self):
        with self.assertRaises(cr.ReportError) as ctx:
            cr.load_settings(public=False, cli_authors="Erika Mustermann")
        self.assertEqual(ctx.exception.status, 2)

    def test_only_qmp_may_approve(self):
        with self.assertRaises(cr.ReportError) as ctx:
            cr.load_settings(public=False, cli_approvers="Student1")
        self.assertEqual(ctx.exception.status, 2)

    def test_committed_config_must_not_hold_local_keys(self):
        self.config.write_text(json.dumps({"lab_url": "https://example.org"}), encoding="utf-8")
        with self.assertRaises(cr.ReportError):
            cr.load_settings(public=True)

    def test_internal_fields_resolve_names(self):
        settings = cr.load_settings(public=False)
        fields = cr.build_person_fields(settings, public=False, build_dir=self.tmp)
        self.assertEqual(fields["AUTHOR"], "Muster")
        self.assertIn("Dr. Erika Mustermann, MSc", fields["CONTACT_BLOCK"])
        self.assertIn(r"\href{mailto:erika@example.org}{erika@example.org}", fields["CONTACT_BLOCK"])
        self.assertIn(r"\href{https://example.org/lab\#x}", fields["FOOTER_URL"])
        self.assertIn("Signature (Muster)", fields["SIGNATURE_BLOCK"])
        self.assertEqual(fields["LOGO"], "")

    def test_public_fields_have_codes_only(self):
        settings = cr.load_settings(public=True)
        fields = cr.build_person_fields(settings, public=True, build_dir=self.tmp)
        self.assertEqual(fields["AUTHOR"], "QMP2")
        self.assertIn("Signature (QMP2)", fields["SIGNATURE_BLOCK"])
        joined = "\n".join(fields.values())
        self.assertNotIn("@", joined.replace("@{}", "").replace("@{\\hspace{1.5cm}}", ""))
        self.assertNotIn("Mustermann", joined)
        self.assertIn("https://example.org/issues", fields["CONTACT_BLOCK"])

    def test_unknown_code_in_internal_build(self):
        settings = cr.load_settings(public=False, cli_authors="QMP9")
        with self.assertRaises(cr.ReportError):
            cr.build_person_fields(settings, public=False, build_dir=self.tmp)

    def test_local_name_tokens(self):
        tokens = cr.local_name_tokens(self.local_config)
        for token in ("Erika", "Mustermann", "Riki", "erika@example.org", "Beispiel"):
            self.assertIn(token, tokens)

    def test_consented_public_name_is_not_a_token(self):
        data = json.loads(self.people.read_text(encoding="utf-8"))
        data["people"][0].update(public_name="E. Mustermann", public_name_consent=True)
        self.people.write_text(json.dumps(data), encoding="utf-8")
        tokens = cr.local_name_tokens(self.local_config)
        self.assertNotIn("Mustermann", tokens)
        self.assertIn("Erika", tokens)

    def test_public_check_needs_the_people_list(self):
        self.local_config.unlink()
        with self.assertRaises(cr.ReportError) as ctx:
            cr.local_name_tokens(self.local_config)
        self.assertEqual(ctx.exception.status, 2)
        self.assertEqual(cr.local_name_tokens(self.local_config, required=False), [])
        self.local_config.write_text(json.dumps({"people_file": str(self.tmp / "missing.json")}), encoding="utf-8")
        with self.assertRaises(cr.ReportError):
            cr.local_name_tokens(self.local_config)

    def test_names_are_split_at_commas(self):
        data = json.loads(self.people.read_text(encoding="utf-8"))
        data["people"][2]["name"] = "Beispiel, Max"
        self.people.write_text(json.dumps(data), encoding="utf-8")
        tokens = cr.local_name_tokens(self.local_config)
        self.assertIn("Beispiel", tokens)
        self.assertNotIn("Beispiel,", tokens)

    def test_approver_role_must_be_qmp(self):
        data = json.loads(self.people.read_text(encoding="utf-8"))
        data["people"][1]["role"] = "Student"
        self.people.write_text(json.dumps(data), encoding="utf-8")
        settings = cr.load_settings(public=False)
        with self.assertRaises(cr.ReportError) as ctx:
            cr.build_person_fields(settings, public=False, build_dir=self.tmp)
        self.assertEqual(ctx.exception.status, 2)

    def test_history_cells_are_escaped(self):
        old = cr.HISTORY_DIR
        cr.HISTORY_DIR = self.tmp
        try:
            (self.tmp / "L9_history.csv").write_text("Year;Max_MAE;Max_RMSE;Max_AbsError\n2025;0.5 mm;1 %;a_b\n",
                                                     encoding="utf-8")
            maxima = {"mae": (0.1, "mm"), "rmse": (0.2, "mm"), "max": (0.3, "mm")}
            table = cr.build_history_table(9, 2026, maxima)
        finally:
            cr.HISTORY_DIR = old
        self.assertIn(r"1 \%", table)
        self.assertIn(r"a\_b", table)


class RenderTest(unittest.TestCase):
    def test_tex_escape(self):
        self.assertEqual(cr.tex_escape("a_b & 50% #1 {x} ~ ^ \\"),
                         r"a\_b \& 50\% \#1 \{x\} \textasciitilde{} \textasciicircum{} \textbackslash{}")

    def test_missing_placeholder_is_fatal_and_values_are_not_rescanned(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            template = tmp / "template"
            template.mkdir()
            (template / "main.tex").write_text("<<A>> <<B>>", encoding="utf-8")
            old = cr.TEMPLATE_DIR
            cr.TEMPLATE_DIR = template
            try:
                build = tmp / "build"
                build.mkdir()
                cr.render_templates(build, {"A": "<<B>>", "B": "b"})
                self.assertEqual((build / "main.tex").read_text(encoding="utf-8"), "<<B>> b")
                build2 = tmp / "build2"
                build2.mkdir()
                with self.assertRaises(cr.ReportError):
                    cr.render_templates(build2, {"A": "a"})
            finally:
                cr.TEMPLATE_DIR = old
        finally:
            shutil.rmtree(tmp)

    def test_template_placeholders_are_all_provided(self):
        root = Path(__file__).resolve().parents[1] / "report" / "template"
        used = set()
        for tex in root.rglob("*.tex"):
            used.update(cr.PLACEHOLDER.findall(tex.read_text(encoding="utf-8")))
        provided = {"LINAC", "MEAS_DATE", "MEAS_TIME", "CREATION_DATE", "CREATION_TIME", "INSTITUTION", "PHANTOM",
                    "OVERALL_PASSRATE", "VERDICT_COLOR", "VERDICT_LABEL", "MAX_ABS_ERROR", "MAX_ABS_ERROR_UNIT",
                    "MAX_MAE", "MAX_RMSE", "TOL_ACCEPT", "TOL_WATCH", "TABLE_PASSRATE_SUMMARY",
                    "TABLE_PASSRATE_DETAIL", "TABLE_ERROR_METRICS", "TABLE_HISTORY", "DESCRIPTION", "PLOT_PAGES",
                    "AUTHOR", "CONTACT_BLOCK", "FOOTER_URL", "LOGO", "SIGNATURE_BLOCK"}
        self.assertEqual(used, provided)

    def test_committed_config_has_codes_only(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "report" / "report_config.json").read_text())
        for code in config["authors"] + config["approvers"] + [config["contact"]]:
            self.assertRegex(code, cr.ROLE_CODE)
        for code in config["approvers"]:
            self.assertRegex(code, cr.QMP_CODE)
        self.assertNotIn("@", json.dumps(config))


if __name__ == "__main__":
    unittest.main()
