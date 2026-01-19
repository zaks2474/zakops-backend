from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from actions.engine.models import ActionError, ActionPayload, ArtifactMetadata, now_utc_iso
from actions.executors._artifacts import resolve_action_artifact_dir
from actions.executors.base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult


class GenerateLoiExecutor(ActionExecutor):
    action_type = "DOCUMENT.GENERATE_LOI"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        # LOI inputs are optional; deal context is required for destination path.
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}

        purchaser_name = str(inputs.get("purchaser_name") or "ZakOps Acquisition LLC").strip()
        target_company = str(inputs.get("target_company") or (ctx.deal or {}).get("canonical_name") or "").strip()
        purchase_price = str(inputs.get("purchase_price") or "[TBD]").strip()
        exclusivity_days = int(inputs.get("exclusivity_days") or 30)
        closing_timeline = str(inputs.get("closing_timeline") or "60 days from LOI").strip()
        notes = str(inputs.get("notes") or "").strip()

        if not target_company:
            target_company = "[TARGET COMPANY]"

        body_lines = [
            "LETTER OF INTENT (LOI)",
            "",
            f"Purchaser: {purchaser_name}",
            f"Target: {target_company}",
            f"Proposed Purchase Price: {purchase_price}",
            f"Exclusivity: {exclusivity_days} days",
            f"Closing Timeline: {closing_timeline}",
            "",
            "This LOI is non-binding except for confidentiality, exclusivity, and related provisions.",
        ]
        if notes:
            body_lines.extend(["", "Notes:", notes])

        out_dir = resolve_action_artifact_dir(ctx)
        artifacts = []

        # Required DOCX
        try:
            from docx import Document  # type: ignore
        except Exception as e:
            raise ActionExecutionError(
                ActionError(
                    code="python_docx_missing",
                    message=f"python-docx is required for LOI generation: {type(e).__name__}",
                    category="dependency",
                    retryable=False,
                )
            )

        doc = Document()
        for line in body_lines:
            doc.add_paragraph(line)

        docx_path = out_dir / "loi.docx"
        doc.save(str(docx_path))
        artifacts.append(
            ArtifactMetadata(
                filename=docx_path.name,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                path=str(docx_path),
                created_at=now_utc_iso(),
            )
        )

        # Optional PDF (best-effort)
        try:
            from reportlab.lib.pagesizes import letter  # type: ignore
            from reportlab.pdfgen import canvas  # type: ignore

            pdf_path = out_dir / "loi.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            width, height = letter
            y = height - 72
            for line in body_lines:
                c.drawString(72, y, line)
                y -= 14
                if y < 72:
                    c.showPage()
                    y = height - 72
            c.save()
            artifacts.append(
                ArtifactMetadata(
                    filename=pdf_path.name,
                    mime_type="application/pdf",
                    path=str(pdf_path),
                    created_at=now_utc_iso(),
                )
            )
        except Exception:
            pass

        outputs: Dict[str, Any] = {
            "purchaser_name": purchaser_name,
            "target_company": target_company,
            "purchase_price": purchase_price,
            "exclusivity_days": exclusivity_days,
            "closing_timeline": closing_timeline,
        }
        return ExecutionResult(outputs=outputs, artifacts=artifacts)

