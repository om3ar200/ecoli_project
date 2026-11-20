from flask import Flask, render_template, jsonify, request
import analysis
import models

app = Flask(__name__)

# الصفحة الرئيسية تعرض الداشبورد
@app.route("/")
def home():
    return render_template("ecoli_dashboard_v2.html")

# مسار الصفحة ثلاثية الأبعاد
@app.route("/3d")
def view_3d():
    return render_template("ecoli_complement_3d.html")

# مسار مثال لتشغيل دالة من analysis.py أو models.py وإرجاع النتيجة
@app.route("/api/analyze", methods=["POST"])
def run_analysis():
    data = request.get_json(silent=True) or {}
    # استبدل run_your_function باسم الدالة الفعلية داخل analysis.py أو models.py
    # result = analysis.run_your_function(data)
    return jsonify({"status": "ready", "received": data})

if __name__ == "__main__":
    app.run(debug=True)
