      <form method="post" enctype="multipart/form-data" id="caseForm">
        <div class="tabs"><button type="button" class="tab active" id="textTab" onclick="showTab('text')">Enter text</button><button type="button" class="tab" id="fileTab" onclick="showTab('file')">Upload document</button></div>
        <div id="textPane"><label for="caseText" class="sub" style="display:block;margin-bottom:8px">Hindi facts and arguments</label><textarea id="caseText" name="case" placeholder="यहाँ जमानत आवेदन से संबंधित तथ्य और तर्क लिखें या पेस्ट करें...">{{ text }}</textarea><div class="hint">For best results, use the facts-and-arguments section. Avoid including the final judgment or outcome.</div></div>
        <div id="filePane" style="display:none"><div class="drop"><div class="glyph">▤</div><strong>Upload a case document</strong><p>PDF, DOCX, TXT, PNG, JPG or JPEG · Maximum {{ max_upload_mb }} MB</p><input class="file-input" type="file" id="document" name="document" accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"><div class="filename" id="fileName"></div></div><div class="hint">Selectable PDFs and Word files are read directly. Scanned PDFs and images require Tesseract OCR with Hindi language data. After extraction, the text is validated before prediction.</div></div>
        <div class="actions"><button class="btn primary" type="submit" name="action" value="predict" id="submitBtn">Validate and analyze</button><button class="btn secondary" type="button" onclick="clearInput()">Clear</button></div>
      </form>
      {% if extracted %}<div style="margin-top:20px"><h2 style="font-size:15px">Extracted text preview</h2><p class="sub">Source: {{ source_method }}{% if source_filename %} · {{ source_filename }}{% endif %}</p><textarea readonly style="min-height:145px">{{ extracted }}</textarea></div>{% endif %}
    </section>
    <aside>
      {% if result %}
      <section class="panel">
        <div class="result-head"><h2>Model result</h2><span class="pill">SCREENED INPUT</span></div>
        {% if result.uncertain %}
          <div class="notice error" style="margin-top:16px;margin-bottom:10px"><strong>Uncertain result — do not rely on a binary outcome.</strong><br>The model's class probabilities are too close under this prototype's conservative display rule. Review the input and consult a qualified legal professional.</div>
        {% else %}
          <div class="outcome {{ 'granted' if result.label == 1 else 'denied' }}">{{ 'BAIL GRANTED' if result.label == 1 else 'BAIL DENIED' }}</div>
          <p class="sub" style="margin-bottom:0">Estimated class from historical text patterns; not a legal conclusion.</p>
        {% endif %}
        <div class="metric-row"><div class="metric"><span>Estimated probability of grant</span><strong>{{ result.prob }}%</strong><div class="bar"><i style="width:{{ result.prob }}%"></i></div></div><div class="metric"><span>Top-class probability*</span><strong>{{ result.conf }}%</strong><div class="bar"><i style="width:{{ result.conf }}%"></i></div></div></div>
        <p class="hint">*This is the classifier's probability, not a calibrated measure of legal certainty.</p>
      </section>
      <section class="panel"><h2>Influential terms</h2><p class="sub">Terms with larger absolute feature contributions influenced this model output. They are not legal reasoning.</p>{% if result.explain %}<ul class="term-list">{% for term, score in result.explain %}<li>{{ term }} <small>({{ '%.3f'|format(score) }})</small></li>{% endfor %}</ul>{% else %}<p class="empty">No influential word features were available for this input.</p>{% endif %}</section>
      <section class="panel"><h2>Similar historical cases</h2><p class="sub">Similarity is lexical and does not prove that cases are legally equivalent.</p>{% if result.similar %}{% for r in result.similar %}<div class="case"><div class="case-title"><span>{{ r.district if r.district else 'District not available' }}</span><span class="pill">{{ 'Granted' if r.y == 1 else 'Denied' }}{% if r.similarity is defined %} · {{ '%.0f'|format(r.similarity * 100) }}% match{% endif %}</span></div><p>{{ (r.text or '')[:650] }}{% if (r.text or '')|length > 650 %}…{% endif %}</p></div>{% endfor %}{% else %}<p class="empty">No sufficiently similar historical cases were found. Do not infer that the prediction is reliable from the absence of matches.</p>{% endif %}</section>
      {% else %}
      <section class="panel"><h2>What this tool checks</h2><p class="sub">The application runs validation before the classifier.</p><div class="case"><div class="case-title"><span>01 · Readability</span><span class="pill">Required</span></div><p>Rejects empty, too-short, repetitive or unreadable text.</p></div><div class="case"><div class="case-title"><span>02 · Legal relevance</span><span class="pill">Required</span></div><p>Looks for bail/legal indicators and text resembling the Hindi legal domain.</p></div><div class="case"><div class="case-title"><span>03 · Prediction screening</span><span class="pill">Cautious</span></div><p>Very close class probabilities are flagged as uncertain. This does not guarantee that other predictions are correct.</p></div></section>
      <section class="panel"><h2>Supported documents</h2><p class="sub">Text extraction is performed locally by the Flask app.</p><ul style="padding-left:19px;font-size:13px;color:#536176"><li>PDF with selectable text</li><li>DOCX and TXT files</li><li>Scanned PDF or image via OCR when dependencies are installed</li></ul></section>
      {% endif %}
    </aside>
  </div>
  <div class="disclaimer"><strong>Research prototype only.</strong> This system estimates patterns from historical Hindi bail data. It does not verify current law, evaluate all case facts, or replace a lawyer or judge. Do not use its output as the basis for a legal decision.</div>
  <div class="footer">LegalLens · Hindi bail-outcome prediction research prototype</div>
</main>
<script>
function showTab(which){const text=which==='text';document.getElementById('textPane').style.display=text?'block':'none';document.getElementById('filePane').style.display=text?'none':'block';document.getElementById('textTab').classList.toggle('active',text);document.getElementById('fileTab').classList.toggle('active',!text);document.getElementById('caseText').required=text;document.getElementById('document').required=!text;}
document.getElementById('document').addEventListener('change',function(){document.getElementById('fileName').textContent=this.files.length?this.files[0].name:'';});
function clearInput(){document.getElementById('caseText').value='';document.getElementById('document').value='';document.getElementById('fileName').textContent='';showTab('text');}
// If a file was selected, ensure the form uses the upload branch. The server
// accepts either pasted text or a selected document.
document.getElementById('caseForm').addEventListener('submit',function(){const file=document.getElementById('document').files.length>0;if(file){document.getElementById('caseText').required=false;}else{document.getElementById('caseText').required=false;}});
</script>
</body>
</html>"""


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    extracted = ""
    source_method = ""
    source_filename = ""
    result = None
    error = None
    info = None

    if request.method == "POST":
        pasted_text = clean(request.form.get("case", ""))
        uploaded = request.files.get("document")

        # If a document is supplied, it is the active input. Otherwise use text.
        if uploaded and uploaded.filename:
            try:
                extracted, source_method, source_filename = extract_document(uploaded)
                text = extracted
                validation_error = validate_case_text(text)
                if validation_error:
                    error = validation_error
                else:
                    result = predict_case(text)
                    info = "Document text was extracted and passed the initial input checks."
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
            except Exception:
                logger.exception("Unexpected document processing error")
                error = "The document could not be processed because of an unexpected error. Check the file format and application logs."
        else:
            text = pasted_text
            validation_error = validate_case_text(text)
            if validation_error:
                error = validation_error
            else:
                try:
                    result = predict_case(text)
                except Exception:
                    logger.exception("Prediction failed")
                    error = "The model could not process this input. No prediction was returned. Check that all trained artifacts are compatible."

    return render_template_string(
        HTML,
        text=text,
        extracted=extracted,
        source_method=source_method,
        source_filename=source_filename,
        result=result,
        error=error,
        info=info,
        max_upload_mb=MAX_UPLOAD_MB,
    )


@app.errorhandler(413)
def too_large(_error):
    return render_template_string(
        HTML,
        text="",
        extracted="",
        source_method="",
        source_filename="",
        result=None,
        error=f"The upload exceeds the {MAX_UPLOAD_MB} MB limit. Please choose a smaller file.",
        info=None,
        max_upload_mb=MAX_UPLOAD_MB,
    ), 413


if __name__ == "__main__":
    # Keep debug disabled unless you are actively debugging on your own machine.
    app.run(host="127.0.0.1", port=5000, debug=False)