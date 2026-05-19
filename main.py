from src.training.ensemble_predictor import EnsemblePredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

predictor = EnsemblePredictor.from_paths()

def multi_campo():
    print("\n── Multi-campo demo ──")
    api_a = {
        "customer_id": 1,
        "text_correo_250": "x@y.com",
        "cp": "06600",
        "nombre_cliente": "Juan Perez",
        "fecha_registro": "2024-01-15",
        "telefono_contacto": "+56912345678",
        "activo_flag": True,
        "monto_total": 150000.50
    }

    api_b = {
        "id_cliente": 1,
        "correo": "x@y.com",
        "codigo_postal": "06600",
        "nombre": "Juan Perez",
        "fecha_alta": "2024-01-15",
        "telefono": "+56912345678",
        "es_activo": True,
        "total_monto": 150000.50
    }
    res = predictor.predecir_objetos(api_a, api_b, estrategia="max")
    print(res)

def evaluar():
    test_data = [
        # ✅ MATCHES VERDADEROS
        {"field_a": "uuu_budget", "field_b": "BUDGET_VALUE", "match": True, "pair": "unifier_sap",
         "domain": "projects"},  # Presupuesto
        {"field_a": "uuu_finish_date", "field_b": "PENDE", "match": True, "pair": "unifier_sap", "domain": "projects"},
        # Fecha fin
        {"field_a": "uuu_phase", "field_b": "PSPHI", "match": True, "pair": "unifier_sap", "domain": "projects"},
        # Fase → Nivel WBS
        {"field_a": "uuu_vendor_id", "field_b": "LIFNR", "match": True, "pair": "unifier_sap", "domain": "projects"},
        # Proveedor
        {"field_a": "uuu_proj_name", "field_b": "POST1", "match": True, "pair": "unifier_sap", "domain": "projects"},
        # Nombre proyecto
        {"field_a": "uuu_commit_amount", "field_b": "WTGBTR", "match": True, "pair": "unifier_sap",
         "domain": "projects"},  # Monto comprometido
        {"field_a": "uuu_proj_id", "field_b": "PS_POSID", "match": True, "pair": "unifier_sap", "domain": "projects"},
        # ID Proyecto Unifier = WBS SAP
        {"field_a": "uuu_actual_cost", "field_b": "ISTWRT", "match": True, "pair": "unifier_sap", "domain": "projects"},
        # Costo real
        {"field_a": "uuu_cost_code", "field_b": "KOSTL", "match": True, "pair": "unifier_sap", "domain": "projects"},
        # Cost code → Centro de costo
        {"field_a": "uuu_start_date", "field_b": "PSTRT", "match": True, "pair": "unifier_sap", "domain": "projects"},
        # Fecha inicio

        # ❌ NO MATCHES
        {"field_a": "uuu_proj_id", "field_b": "LIFNR", "match": False, "pair": "unifier_sap", "domain": "projects"},
        {"field_a": "uuu_actual_cost", "field_b": "POST1", "match": False, "pair": "unifier_sap", "domain": "projects"},
        {"field_a": "uuu_vendor_id", "field_b": "PSTRT", "match": False, "pair": "unifier_sap", "domain": "projects"},
        {"field_a": "uuu_budget", "field_b": "AUFNR", "match": False, "pair": "unifier_sap", "domain": "projects"},
        {"field_a": "uuu_forecast", "field_b": "PSPNR", "match": False, "pair": "unifier_sap", "domain": "projects"},

        # ⚠️  CASOS AMBIGUOS — requieren revisión manual
        {"field_a": "uuu_status", "field_b": "OBJNR", "match": False, "pair": "unifier_sap", "domain": "projects"},
        # Status ≠ object number SAP
        {"field_a": "uuu_contract_no", "field_b": "AUFNR", "match": False, "pair": "unifier_sap", "domain": "projects"},
        # Contrato Unifier ≠ Orden SAP (pero se usan igual)

    ]

    y_true = []
    y_pred = []
    y_scores = []

    predicciones_detalladas = []

    for item in test_data:
        c1 = item["field_a"]
        c2 = item["field_b"]
        esperado = item["match"]

        resultado = predictor.predecir(c1, c2)
        predicciones_detalladas.append(resultado)
        y_true.append(1 if esperado else 0)  # Convert boolean to int for consistency
        if isinstance(resultado, dict):
            pred = 1 if resultado.get('son_similares', False) else 0
            score = resultado.get('score_final', resultado.get('porcentaje', 0))
            if isinstance(score, str):
                score = float(score.strip('%')) / 100
        else:
            pred = int(resultado) if isinstance(resultado, (int, float)) else 0
            score = float(pred)

        y_pred.append(pred)
        print(f"resultado: {resultado}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        square=True,
        linewidths=1,
        linecolor='white',
        annot_kws={'size': 16, 'weight': 'bold'},
        xticklabels=['No Match (0)', 'Match (1)'],  # ← fix 3: labels en heatmap, no después
        yticklabels=['No Match (0)', 'Match (1)'],
    )

    ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
    ax.set_title('Matriz de Confusión - Semantic Field Matcher', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    print(f"accuracy={accuracy:.3f} | precision={precision:.3f} | recall={recall:.3f} | f1={f1:.3f}")
    print("\n" + classification_report(y_true, y_pred, target_names=["No Match", "Match"], zero_division=0))
if __name__ == '__main__':
    evaluar()