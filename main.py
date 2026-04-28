from src.training.ensemble_predictor import EnsemblePredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import json
import keras
import src.config as config
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_training.gen_datos import generacion_datos

predictor = EnsemblePredictor.from_paths(
    model_path="src/training/models/v3.2.1/semantic_field_matcher_v1.h5"
)

def multi_campo():
    print("\n── Multi-campo demo ──")
    api_a = {"customer_id": 1, "text_correo_250": "x@y.com", "cp": "06600"}
    api_b = {"id_cliente": 1, "correo": "x@y.com", "codigo_postal": "06600"}
    res = predictor.predecir_objetos(api_a, api_b, estrategia="top3")
    print(res)

def evaluar():
    # train_data, test_data = generacion_datos(test_size=0.2, random_state=42)
    test_data =[
        # ✅ MATCHES VERDADEROS — misma semántica, diferente nomenclatura
        {"field_a": "customer_id", "field_b": "client_id", "match": True},
        {"field_a": "customer_id", "field_b": "cust_id", "match": True},
        {"field_a": "customer_id", "field_b": "clienteId", "match": True},
        {"field_a": "first_name", "field_b": "nombre", "match": True},
        {"field_a": "first_name", "field_b": "fname", "match": True},
        {"field_a": "last_name", "field_b": "apellido", "match": True},
        {"field_a": "last_name", "field_b": "surname", "match": True},
        {"field_a": "email", "field_b": "email_address", "match": True},
        {"field_a": "email", "field_b": "correo_electronico", "match": True},
        {"field_a": "phone", "field_b": "phone_number", "match": True},
        {"field_a": "phone", "field_b": "telefono", "match": True},
        {"field_a": "phone", "field_b": "mobile", "match": True},
        {"field_a": "date_of_birth", "field_b": "dob", "match": True},
        {"field_a": "date_of_birth", "field_b": "birth_date", "match": True},
        {"field_a": "date_of_birth", "field_b": "fecha_nacimiento", "match": True},
        {"field_a": "created_at", "field_b": "creation_date", "match": True},
        {"field_a": "created_at", "field_b": "fecha_creacion", "match": True},
        {"field_a": "updated_at", "field_b": "last_modified", "match": True},
        {"field_a": "updated_at", "field_b": "modification_date", "match": True},
        {"field_a": "street_address", "field_b": "address_line_1", "match": True},
        {"field_a": "street_address", "field_b": "direccion", "match": True},
        {"field_a": "zip_code", "field_b": "postal_code", "match": True},
        {"field_a": "zip_code", "field_b": "codigo_postal", "match": True},
        {"field_a": "country", "field_b": "pais", "match": True},
        {"field_a": "country_code", "field_b": "iso_country", "match": True},
        {"field_a": "gender", "field_b": "sexo", "match": True},
        {"field_a": "gender", "field_b": "genero", "match": True},
        {"field_a": "total_amount", "field_b": "monto_total", "match": True},
        {"field_a": "total_amount", "field_b": "grand_total", "match": True},
        {"field_a": "total_amount", "field_b": "importe_total", "match": True},
        {"field_a": "unit_price", "field_b": "precio_unitario", "match": True},
        {"field_a": "unit_price", "field_b": "price_per_unit", "match": True},
        {"field_a": "quantity", "field_b": "qty", "match": True},
        {"field_a": "quantity", "field_b": "cantidad", "match": True},
        {"field_a": "discount", "field_b": "descuento", "match": True},
        {"field_a": "tax_amount", "field_b": "iva", "match": True},
        {"field_a": "tax_amount", "field_b": "impuesto", "match": True},
        {"field_a": "order_id", "field_b": "pedido_id", "match": True},
        {"field_a": "order_id", "field_b": "purchase_id", "match": True},
        {"field_a": "invoice_number", "field_b": "numero_factura", "match": True},
        {"field_a": "invoice_date", "field_b": "fecha_factura", "match": True},
        {"field_a": "due_date", "field_b": "fecha_vencimiento", "match": True},
        {"field_a": "payment_method", "field_b": "metodo_pago", "match": True},
        {"field_a": "payment_status", "field_b": "estado_pago", "match": True},
        {"field_a": "is_active", "field_b": "active", "match": True},
        {"field_a": "is_active", "field_b": "activo", "match": True},
        {"field_a": "is_deleted", "field_b": "deleted", "match": True},
        {"field_a": "product_name", "field_b": "nombre_producto", "match": True},
        {"field_a": "product_code", "field_b": "sku", "match": True},
        {"field_a": "product_code", "field_b": "codigo_producto", "match": True},
        {"field_a": "category", "field_b": "categoria", "match": True},
        {"field_a": "description", "field_b": "descripcion", "match": True},
        {"field_a": "stock", "field_b": "inventory", "match": True},
        {"field_a": "stock", "field_b": "existencias", "match": True},
        {"field_a": "employee_id", "field_b": "emp_id", "match": True},
        {"field_a": "employee_id", "field_b": "staff_id", "match": True},
        {"field_a": "salary", "field_b": "sueldo", "match": True},
        {"field_a": "hire_date", "field_b": "fecha_contratacion", "match": True},
        {"field_a": "department", "field_b": "departamento", "match": True},
        {"field_a": "manager_id", "field_b": "supervisor_id", "match": True},
        {"field_a": "company_name", "field_b": "razon_social", "match": True},
        {"field_a": "company_name", "field_b": "empresa", "match": True},
        {"field_a": "tax_id", "field_b": "rut", "match": True},
        {"field_a": "tax_id", "field_b": "nit", "match": True},
        {"field_a": "tax_id", "field_b": "rfc", "match": True},
        {"field_a": "ip_address", "field_b": "ip", "match": True},
        {"field_a": "user_agent", "field_b": "browser_info", "match": True},
        {"field_a": "session_id", "field_b": "session_token", "match": True},
        {"field_a": "longitude", "field_b": "lng", "match": True},
        {"field_a": "latitude", "field_b": "lat", "match": True},
        {"field_a": "notes", "field_b": "comentarios", "match": True},
        {"field_a": "notes", "field_b": "observaciones", "match": True},
        {"field_a": "attachment", "field_b": "archivo_adjunto", "match": True},
        {"field_a": "url", "field_b": "link", "match": True},
        {"field_a": "url", "field_b": "website", "match": True},

        # ❌ NO MATCHES — campos distintos, algunos con nombres engañosos
        {"field_a": "customer_id", "field_b": "vendor_id", "match": False},
        {"field_a": "customer_id", "field_b": "product_id", "match": False},
        {"field_a": "customer_id", "field_b": "order_id", "match": False},
        {"field_a": "first_name", "field_b": "last_name", "match": False},
        {"field_a": "first_name", "field_b": "full_name", "match": False},  # parte vs. todo
        {"field_a": "email", "field_b": "phone", "match": False},
        {"field_a": "email", "field_b": "username", "match": False},        # relacionado pero distinto
        {"field_a": "date_of_birth", "field_b": "hire_date", "match": False},
        {"field_a": "date_of_birth", "field_b": "created_at", "match": False},
        {"field_a": "created_at", "field_b": "deleted_at", "match": False},
        {"field_a": "street_address", "field_b": "zip_code", "match": False},
        {"field_a": "street_address", "field_b": "city", "match": False},
        {"field_a": "city", "field_b": "country", "match": False},
        {"field_a": "total_amount", "field_b": "unit_price", "match": False},
        {"field_a": "total_amount", "field_b": "discount", "match": False},
        {"field_a": "quantity", "field_b": "unit_price", "match": False},
        {"field_a": "tax_amount", "field_b": "discount", "match": False},
        {"field_a": "invoice_number", "field_b": "order_id", "match": False},  # tricky — ambos son IDs de transacción
        {"field_a": "payment_method", "field_b": "payment_status", "match": False},
        {"field_a": "is_active", "field_b": "is_deleted", "match": False},
        {"field_a": "product_name", "field_b": "category", "match": False},
        {"field_a": "product_code", "field_b": "product_name", "match": False},
        {"field_a": "stock", "field_b": "quantity", "match": False},       # tricky — stock ≠ qty del pedido
        {"field_a": "employee_id", "field_b": "manager_id", "match": False},
        {"field_a": "salary", "field_b": "tax_amount", "match": False},
        {"field_a": "department", "field_b": "category", "match": False},
        {"field_a": "company_name", "field_b": "product_name", "match": False},
        {"field_a": "tax_id", "field_b": "customer_id", "match": False},
        {"field_a": "latitude", "field_b": "longitude", "match": False},   # coordenadas distintas
        {"field_a": "session_id", "field_b": "user_id", "match": False},
        {"field_a": "url", "field_b": "email", "match": False},
        {"field_a": "notes", "field_b": "description", "match": False},    # tricky — semánticamente cercanos
        {"field_a": "attachment", "field_b": "url", "match": False},       # tricky — ambos son referencias a recursos
        {"field_a": "gender", "field_b": "date_of_birth", "match": False},
        {"field_a": "zip_code", "field_b": "country_code", "match": False},
        {"field_a": "ip_address", "field_b": "session_id", "match": False},
        {"field_a": "due_date", "field_b": "hire_date", "match": False},
        {"field_a": "invoice_date", "field_b": "due_date", "match": False},
        {"field_a": "phone", "field_b": "fax", "match": False},            # tricky — ambos son de contacto
        {"field_a": "username", "field_b": "employee_id", "match": False},
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





