import json
import random
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.version_utils import training
import src.db.config_mongodb as config_mongodb
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

_db = config_mongodb.SemanticConfigMongoDb()

class generadorDatos:
    def __init__(self):
        self.conceptos_base = {
            'personales': [
                'nombre', 'apellido', 'apellido_paterno', 'apellido_materno',
                'edad', 'genero', 'sexo', 'fecha_nacimiento', 'lugar_nacimiento',
                'nacionalidad', 'estado_civil', 'profesion', 'ocupacion',
                'titulo', 'sufijo_nombre', 'nombre_completo'
            ],
            'contacto': [
                'email', 'email_alternativo', 'email_principal',
                'telefono', 'telefono_casa', 'telefono_oficina', 'telefono_movil',
                'celular', 'fax', 'whatsapp', 'telegram', 'skype', 'zoom',
                'sitio_web', 'linkedin', 'red_social'
            ],
            'fiscales': [
                'rfc', 'curp', 'uuid', 'cfdi', 'uso_cfdi', 'uso_cfdi_descripcion',
                'regimen_fiscal', 'regimen_fiscal_descripcion', 'tipo_persona',
                'id_fiscal', 'nss', 'infonavit', 'tipo_contribuyente',
                'retenciones', 'comprobante_fiscal', 'serie_factura',
                'folio_factura', 'certificado_sat', 'no_certificado'
            ],
            'domicilio': [
                'calle', 'numero_exterior', 'numero_interior', 'numero_int',
                'colonia', 'barrio', 'ciudad', 'poblacion', 'municipio',
                'delegacion', 'alcaldia', 'estado', 'provincia', 'region',
                'pais', 'codigo_postal', 'cp', 'entre_calles', 'referencia',
                'latitud', 'longitud', 'zona', 'domicilio_completo',
                'tipo_asentamiento', 'clave_oficina'
            ],
            'comerciales': [
                'razon_social', 'nombre_comercial', 'nombre_fantasia',
                'giro', 'actividad_economica', 'sector', 'tipo_empresa',
                'empresa', 'sucursal', 'division', 'departamento',
                'clasificacion', 'categoria', 'segmento', 'origen',
                'tipo_cliente', 'tipo_proveedor', 'estatus_cliente'
            ],
            'facturacion': [
                'metodo_pago', 'forma_pago', 'moneda', 'tasa_iva', 'tasa_isr',
                'tipo_cambio', 'descuento', 'plazo_pago', 'dias_credito',
                'limite_credito', 'saldo', 'cuenta_bancaria', 'clabe',
                'banco', 'referencia_pago', 'condiciones_pago',
                'impuestos', 'subtotal', 'total', 'tasa_ieps', 'cuenta_contable'
            ],
            'empresariales_erp': [
                'codigo_cliente', 'codigo_proveedor', 'codigo_producto',
                'sku', 'upc', 'ean', 'part_number', 'modelo',
                'existencia', 'stock_minimo', 'stock_maximo', 'ubicacion_almacen',
                'precio_compra', 'precio_venta', 'costo_promedio', 'costo_estandar',
                'lote', 'serie', 'fecha_caducidad', 'almacen', 'racks'
            ],
            'transacciones': [
                'folio', 'folio_orden', 'folio_factura', 'folio_remision',
                'folio_pedido', 'folio_cotizacion', 'serie', 'fecha_emision',
                'fecha_creacion', 'fecha_modificacion', 'fecha_cierre',
                'usuario_creacion', 'usuario_modificacion', 'estado',
                'estatus', 'autorizacion', 'referencia', 'observaciones',
                'notas_internas', 'notas_cliente'
            ],
            'bancarios': [
                'cuenta_contable', 'cuenta_bancaria', 'clabe', 'iban', 'swift',
                'numero_cuenta', 'tarjeta', 'tipo_cuenta', 'banco_nombre',
                'banco_emisor', 'sucursal_bancaria', 'moneda_cuenta'
            ],
            'logisticos': [
                'metodo_envio', 'transportista', 'guia', 'tracking_number',
                'peso', 'volumen', 'dimensiones', 'embalaje', 'tipo_envio',
                'fecha_envio', 'fecha_entrega', 'hora_entrega', 'recibe',
                'ruta', 'zona_entrega', 'costo_envio'
            ]
        }

        self.prefijos_comunes = [
            'cust', 'custrecord', 'cf', 'inv', 'tx', 'bill', 'ship',
            'main', 'alt', 'secondary', 'primary', 'work', 'home',
            'cli', 'prov', 'prod', 'art', 'serv', 'doc', 'emp',
            'fin', 'cont', 'com', 'log', 'alm', 'ventas', 'compras',
            'rh', 'nomina', 'teso', 'cxc', 'cxp', 'inv', 'act', 'pasivo',
            'cap', 'ing', 'egr', 'cta', 'sub', 'aux', 'sys', 'tmp'
        ]

        self.sufijos_comunes = [
            '_id', '_code', '_name', '_number', '_desc', '_value',
            '_amount', '_date', '_time', '_type', '_status', '_flag',
            '_key', '_ref', '_text', '_longtext', '_decimal', '_int',
            '_bool', '_dt', '_timestamp', '_month', '_year', '_periodo',
            '_folio', '_serie', '_uuid', '_rfc', '_curp', '_email',
            '_tel', '_dir', '_cp', '_cta', '_porc', '_total'
        ]

    def generar_pares_positivos(self, concepto, variaciones=5):
        pares = []
        variantes = self._generar_variantes(concepto, variaciones)

        for i in range(len(variantes)):
            for j in range(i + 1, len(variantes)):
                pares.append((variantes[i], variantes[j], 1))

        return pares

    def generar_pares_negativos(self, concepto1, concepto2, variaciones=3):
        pares = []
        variantes1 = self._generar_variantes(concepto1, variaciones)
        variantes2 = self._generar_variantes(concepto2, variaciones)

        for v1 in variantes1:
            for v2 in variantes2:
                pares.append((v1, v2, 0))

        return pares

    def _generar_variantes(self, palabra_base, num_variaciones):
        variantes = [palabra_base]

        for prefijo in random.sample(self.prefijos_comunes, min(num_variaciones, len(self.prefijos_comunes))):
            variantes.append(f"{prefijo}_{palabra_base}")

        for sufijo in random.sample(self.sufijos_comunes, min(num_variaciones, len(self.sufijos_comunes))):
            variantes.append(f"{palabra_base}{sufijo}")

        variantes.append(palabra_base.upper())
        variantes.append(palabra_base.lower())
        variantes.append(palabra_base.replace('_', ''))

        return list(set(variantes))[:num_variaciones * 2]

def cargar_pares(filepath=None):
    if filepath is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        filepath = project_root / "src" / "data_training" / "datos_entrenamiento" / "training_data_23-03.jsonl"
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generacion_datos(test_size=0.2, random_state=42):
    generador = generadorDatos()
    datos_ent = []

    # Intentar cargar desde MongoDB; si falla, cargar desde archivos
    try:
        datos = _db.cargar_dataset(
            nombre="dataset_alpha",
            version="1.1",
            return_dicts=True
        )
        training_data = [(p["field_a"], p["field_b"], p["match"]) for p in datos]
        logger.info(f"Cargados {len(training_data)} pares desde MongoDB")
    except Exception:
        training_data = cargar_pares()
        training_data = [(p["field_a"], p["field_b"], p["match"]) for p in training_data]
        logger.info(f"Cargados {len(training_data)} pares desde archivos JSONL")

    datos_ent.extend(training_data)

    conceptos_clave = ['nombre', 'email', 'telefono', 'rfc', 'uuid', 'direccion', 'cp']

    for concepto in conceptos_clave:
        datos_ent.extend(generador.generar_pares_positivos(concepto, variaciones=8))

    # FIX: limit negative pairs to avoid severe class imbalance
    # Instead of all combinations, sample a balanced number
    conceptos = list(generador.conceptos_base.keys())
    pares_negativos = []
    for i in range(len(conceptos)):
        for j in range(i + 1, len(conceptos)):
            concepto1 = random.choice(generador.conceptos_base[conceptos[i]])
            concepto2 = random.choice(generador.conceptos_base[conceptos[j]])
            pares_negativos.extend(generador.generar_pares_negativos(concepto1, concepto2, variaciones=2))

    # Count positives first, then balance
    positivos = [d for d in datos_ent if d[2] == 1]
    # Sample negatives to be at most 2x positives (2:1 ratio)
    max_negativos = len(positivos) * 2
    random.shuffle(pares_negativos)
    pares_negativos = pares_negativos[:max_negativos]
    datos_ent.extend(pares_negativos)
    datos_ent = list(set(datos_ent))

    positivos_final = sum(1 for d in datos_ent if d[2] == 1)
    negativos_final = sum(1 for d in datos_ent if d[2] == 0)
    print(f"[OK] Dataset total: {len(datos_ent)} pares | Positivos: {positivos_final} | Negativos: {negativos_final}")

    # DIVISIÓN 80% ENTRENAMIENTO / 20% PRUEBA
    # Separar características (X) y etiquetas (y)
    X = [(c1, c2) for c1, c2, _ in datos_ent]
    y = [match for _, _, match in datos_ent]

    # División estratificada para mantener proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Mantiene la misma proporción de positivos/negativos
    )

    # Reconstruir datasets en formato original
    train_data = [(c1, c2, label) for (c1, c2), label in zip(X_train, y_train)]
    test_data = [(c1, c2, label) for (c1, c2), label in zip(X_test, y_test)]


    return train_data, test_data