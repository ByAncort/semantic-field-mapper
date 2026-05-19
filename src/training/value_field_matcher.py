"""
value_field_matcher.py
──────────────────────
Matching entre campos source y target combinando:
  1. Similitud por NOMBRE (campo semantico existente del ensemble)
  2. Similitud por VALOR  (comparacion directa de valores)

Formula:
  score_total = 0.4 * sim_nombre + 0.6 * sim_valor

Umbral de match: 0.7
Casos especiales:
  - Valor exacto => match forzado
  - Nombre igual pero valor distinto => penalizacion
"""

from __future__ import annotations

import re
import logging
from difflib import SequenceMatcher
from datetime import datetime
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────

PESO_NOMBRE = 0.4
PESO_VALOR = 0.6
UMBRAL_MATCH = 0.7
UMBRAL_EXACTO_VALOR = 0.95  # si la similitud de valor >= esto, match forzado

# Patrones de deteccion de tipo
RE_RUT = re.compile(r'^\d{1,2}\.\d{3}\.\d{3}[\-–−]?\d?[\dkK]$', re.IGNORECASE)
RE_RUT_SIMPLE = re.compile(r'^\d{7,8}[\-–−]?\d?[\dkK]$', re.IGNORECASE)
RE_RUT_NO_DOTS = re.compile(r'^\d{7,9}[\-–−]?[\dkK]$', re.IGNORECASE)
RE_DATE = re.compile(
    r'^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$'
    r'|^(\d{2,4})[/-](\d{1,2})[/-](\d{1,2})$'
    r'|^(\d{4})-(\d{2})-(\d{2})$'
    r'|^(\d{1,2})-(\d{1,2})-(\d{4})$'
)
RE_EMAIL = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
RE_PHONE = re.compile(r'^\+?[\d\s\-\(\)]{7,15}$')
RE_NUMERIC = re.compile(r'^[+-]?\d*\.?\d+$')


# ─────────────────────────────────────────────────────────────────────────────
# DETECCION DE TIPO DE VALOR
# ─────────────────────────────────────────────────────────────────────────────

def detect_value_type(val: str) -> str:
    """
    Detecta el tipo de un valor string.
    Retorna: 'rut', 'email', 'phone', 'date', 'number', 'string'
    """
    s = val.strip()
    if not s:
        return 'empty'

    # RUT: probar primero (tiene prioridad sobre phone/date)
    if RE_RUT.match(s) or RE_RUT_SIMPLE.match(s) or RE_RUT_NO_DOTS.match(s):
        return 'rut'

    # Email
    if RE_EMAIL.match(s):
        return 'email'

    # Date: probar antes que phone (fechas como 15-03-1985 parecen telefonos)
    if RE_DATE.match(s) and parse_date(s) is not None:
        return 'date'

    # Phone: solo si tiene + o parentesis, o tiene muchos digitos
    digits = re.sub(r'\D', '', s)
    if RE_PHONE.match(s) and len(digits) >= 8:
        return 'phone'

    # Number
    if RE_NUMERIC.match(s):
        return 'number'

    return 'string'


# ─────────────────────────────────────────────────────────────────────────────
# SIMILITUD POR VALOR
# ─────────────────────────────────────────────────────────────────────────────

def normalize_rut(rut: str) -> str:
    """
    Normaliza un RUT a formato canonico: '12345678-K'
    Maneja: '12345678-9', '12.345.678-9', '123456789', '76.123.456-K'
    """
    s = rut.strip().upper()
    # Eliminar puntos y espacios
    s = s.replace('.', '').replace(' ', '')
    # Separar cuerpo y digito verificador
    if '-' in s:
        parts = s.split('-')
        cuerpo = parts[0].replace('-', '')
        dv = parts[1]
    else:
        cuerpo = s[:-1]
        dv = s[-1]
    return cuerpo + '-' + dv


def parse_date(s: str) -> datetime | None:
    """
    Intenta parsear una fecha en multiples formatos.
    """
    formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
        '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%d-%m-%y', '%m-%d-%y',
        '%Y.%m.%d', '%d.%m.%Y',
    ]
    s = s.strip()
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def normalize_string(s: str) -> str:
    """
    Normaliza un string para comparacion:
    lowercase, sin acentos, sin espacios extra.
    """
    import unicodedata
    s = s.strip().lower()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+', ' ', s)
    return s


def sim_value(val1: Any, val2: Any) -> tuple[float, str]:
    """
    Calcula similitud entre dos valores.
    Retorna: (score_0_1, justificacion)
    """
    s1 = str(val1).strip()
    s2 = str(val2).strip()

    if not s1 or not s2:
        return 0.0, "valores vacios"

    # Match exacto string
    if s1 == s2:
        return 1.0, "coincidencia exacta"

    # Detectar tipos
    t1 = detect_value_type(s1)
    t2 = detect_value_type(s2)

    # Ambos RUT
    if t1 == 'rut' and t2 == 'rut':
        return 1.0 if normalize_rut(s1) == normalize_rut(s2) else 0.0, "comparacion RUT"

    # Ambos fechas
    if t1 == 'date' and t2 == 'date':
        d1 = parse_date(s1)
        d2 = parse_date(s2)
        if d1 and d2:
            return 1.0 if d1 == d2 else 0.0, "comparacion fechas"

    # Ambos emails
    if t1 == 'email' and t2 == 'email':
        return 1.0 if s1.lower() == s2.lower() else 0.0, "comparacion emails"

    # Ambos telefonos
    if t1 == 'phone' and t2 == 'phone':
        digits1 = re.sub(r'\D', '', s1)
        digits2 = re.sub(r'\D', '', s2)
        # Permitir diferencia de prefijo pais
        if digits1[-7:] == digits2[-7:]:
            return 0.95, "telefonos coinciden en numero local"
        return 0.0, "telefonos distintos"

    # Ambos numeros
    if t1 == 'number' and t2 == 'number':
        try:
            n1 = float(s1)
            n2 = float(s2)
            if n1 == 0 and n2 == 0:
                return 1.0, "numeros iguales"
            max_abs = max(abs(n1), abs(n2))
            diff = abs(n1 - n2) / max_abs
            return max(0.0, 1.0 - diff), "comparacion numerica"
        except ValueError:
            pass

    # Tipos incompatibles => penalizar
    if t1 != t2 and t1 not in ('string', 'empty') and t2 not in ('string', 'empty'):
        return 0.0, f"tipos incompatibles ({t1} vs {t2})"

    # Strings generales: similitud de texto
    norm1 = normalize_string(s1)
    norm2 = normalize_string(s2)

    if norm1 == norm2:
        return 1.0, "strings normalizados iguales"

    seq_sim = SequenceMatcher(None, norm1, norm2).ratio()

    # Token-based Jaccard
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    if tokens1 and tokens2:
        jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        sim = 0.5 * seq_sim + 0.5 * jaccard
    else:
        sim = seq_sim

    reason = f"similitud de texto (seq={seq_sim:.2f})"
    return sim, reason


# ─────────────────────────────────────────────────────────────────────────────
# SIMILITUD POR NOMBRE
# ─────────────────────────────────────────────────────────────────────────────

# Semantic groups for name matching (from project's MongoDB config)
SEMANTIC_GROUPS = {
    # Identifiers
    'id': {'id', 'identifier', 'code', 'key', 'num', 'numero'},
    'customer': {'customer', 'client', 'cliente', 'cust'},
    'name': {'name', 'nombre', 'fname', 'lname', 'full', 'first', 'last', 'apellido'},
    'email': {'email', 'correo', 'mail', 'e-mail'},
    'phone': {'phone', 'telefono', 'tel', 'mobile', 'celular', 'fax'},
    'address': {'address', 'direccion', 'street', 'addr', 'domicilio'},
    'date': {'date', 'fecha', 'dob', 'birth', 'nac', 'created', 'updated', 'modified', 'hire'},
    'amount': {'amount', 'monto', 'total', 'price', 'precio', 'cost', 'costo', 'value', 'valor'},
    'status': {'status', 'estado', 'activo', 'active', 'deleted', 'is'},
    'tax': {'tax', 'tax_id', 'rut', 'nit', 'rfc', 'iva', 'impuesto'},
    'quantity': {'quantity', 'qty', 'cantidad', 'stock', 'inventory', 'existencias'},
    'product': {'product', 'producto', 'sku', 'item', 'good'},
    'order': {'order', 'pedido', 'purchase', 'compra', 'invoice', 'factura'},
    'payment': {'payment', 'pago', 'method', 'metodo', 'status'},
    'company': {'company', 'empresa', 'razon', 'social', 'business'},
    'employee': {'employee', 'emp', 'staff', 'empleado'},
    'department': {'department', 'dept', 'departamento', 'area'},
    'category': {'category', 'categoria', 'type', 'tipo', 'class', 'clase'},
    'description': {'description', 'descripcion', 'desc', 'notes', 'notas', 'comments', 'comentarios'},
    'url': {'url', 'link', 'website', 'web', 'site', 'href'},
    'file': {'file', 'archivo', 'attachment', 'adjunto', 'document', 'documento'},
    'ip': {'ip', 'address_ip', 'ip_address', 'host'},
    'session': {'session', 'token', 'sid', 'uuid', 'guid'},
    'location': {'city', 'ciudad', 'country', 'pais', 'region', 'state', 'zip', 'postal', 'codigo'},
    'gender': {'gender', 'sexo', 'genero', 'sex'},
    'salary': {'salary', 'sueldo', 'wage', 'compensation'},
    'manager': {'manager', 'supervisor', 'jefe', 'boss'},
    'notes': {'notes', 'notas', 'comments', 'comentarios', 'observaciones', 'observations'},
}

# Reverse index: token -> group name
TOKEN_TO_GROUP: dict[str, str] = {}
for group, tokens in SEMANTIC_GROUPS.items():
    for token in tokens:
        TOKEN_TO_GROUP[token] = group


def sim_nombre(nombre1: str, nombre2: str) -> tuple[float, str]:
    """
    Calcula similitud entre dos nombres de campo.
    Usa reglas deterministicas + grupos semanticos del proyecto.
    Retorna: (score_0_1, justificacion)
    """
    n1 = nombre1.strip().lower()
    n2 = nombre2.strip().lower()

    if n1 == n2:
        return 1.0, "nombres identicos"

    # Tokenizacion (misma logica que cross_validator.py)
    def tokenizar(texto: str) -> set:
        texto = texto.lower()
        texto = re.sub(r'[.\-/\\]', '_', texto)
        texto = re.sub(r'([a-z])([A-Z])', r'\1_\2', texto)
        texto = re.sub(r'([a-zA-Z])(\d)', r'\1_\2', texto)
        texto = re.sub(r'(\d)([a-zA-Z])', r'\1_\2', texto)
        tokens = re.split(r'[^a-z0-9]+', texto)
        return {t for t in tokens if t and len(t) >= 2}

    t1 = tokenizar(nombre1)
    t2 = tokenizar(nombre2)

    if not t1 or not t2:
        return 0.0, "sin tokens"

    # Semantic group matching
    groups1 = {TOKEN_TO_GROUP.get(tok) for tok in t1 if tok in TOKEN_TO_GROUP}
    groups2 = {TOKEN_TO_GROUP.get(tok) for tok in t2 if tok in TOKEN_TO_GROUP}
    groups1.discard(None)
    groups2.discard(None)
    group_overlap = len(groups1 & groups2)
    group_max = max(len(groups1 | groups2), 1)
    semantic_sim = group_overlap / group_max if (groups1 or groups2) else 0.0

    # Jaccard de tokens
    inter = t1 & t2
    union = t1 | t2
    jaccard = len(inter) / len(union)

    # Sequence similarity
    seq_sim = SequenceMatcher(None, n1, n2).ratio()

    # Prefix similarity
    prefix_sim = SequenceMatcher(None, n1[:3], n2[:3]).ratio()

    # Penalizacion por tokens diferenciadores
    tokens_dif = {'customer', 'vendor', 'client', 'billing', 'shipping',
                  'primary', 'secondary', 'home', 'work', 'min', 'max',
                  'paterno', 'materno', 'venta', 'costo', 'creacion',
                  'modificacion', 'factura', 'pedido', 'first', 'last', 'full'}
    dif1 = t1 - t2
    dif2 = t2 - t1
    pen = 0.0
    if (dif1 & tokens_dif) and (dif2 & tokens_dif):
        pen = 0.4
    elif (dif1 & tokens_dif) or (dif2 & tokens_dif):
        pen = 0.2

    # Si los grupos semanticos coinciden, boost al score
    if semantic_sim > 0.5:
        boost = semantic_sim * 0.3
    else:
        boost = 0.0

    # Score compuesto
    raw = (
        0.35 * jaccard
        + 0.20 * seq_sim
        + 0.10 * prefix_sim
        + 0.25 * semantic_sim
        + 0.10 * (1.0 if inter else 0.0)
        + boost
        - pen
    )
    score = max(0.0, min(1.0, raw))

    # Justificacion
    if groups1 & groups2:
        reason = f"mismo grupo semantico: {groups1 & groups2}"
    elif jaccard > 0.5:
        reason = f"tokens compartidos: {inter}"
    elif seq_sim > 0.7:
        reason = "similitud secuencial alta"
    elif inter:
        reason = f"tokens parciales: {inter}"
    elif semantic_sim > 0:
        reason = f"grupos semanticos relacionados: {groups1} vs {groups2}"
    else:
        reason = "nombres distintos"

    return score, reason


# ─────────────────────────────────────────────────────────────────────────────
# MATCHING PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldMatch:
    """Resultado de un match entre campos."""
    source_field: str
    target_field: str
    source_value: Any
    target_value: Any
    score_nombre: float
    score_valor: float
    score_total: float
    is_match: bool
    justificacion_nombre: str
    justificacion_valor: str
    justificacion_final: str


def match_fields(
    source: dict,
    target: dict,
    peso_nombre: float = PESO_NOMBRE,
    peso_valor: float = PESO_VALOR,
    umbral: float = UMBRAL_MATCH,
) -> list[FieldMatch]:
    """
    Realiza matching entre todos los pares de campos source/target.

    Args:
        source: Diccionario source {campo: valor}
        target: Diccionario target {campo: valor}
        peso_nombre: Peso para similitud de nombre (default 0.4)
        peso_valor: Peso para similitud de valor (default 0.6)
        umbral: Umbral minimo para considerar match (default 0.7)

    Returns:
        Lista de FieldMatch ordenada por score_total descendente.
    """
    results = []

    for src_field, src_val in source.items():
        best_match = None
        best_score = -1.0

        for tgt_field, tgt_val in target.items():
            # Calcular similitudes
            s_nombre, just_nombre = sim_nombre(src_field, tgt_field)
            s_valor, just_valor = sim_value(src_val, tgt_val)

            # Formula principal
            score_total = (peso_nombre * s_nombre) + (peso_valor * s_valor)

            # Caso especial: valor exacto => forzar match
            if s_valor >= UMBRAL_EXACTO_VALOR:
                score_total = max(score_total, 0.85)
                just_valor += " => match forzado por valor"

            # Caso especial: nombre igual pero valor muy distinto => penalizar
            if s_nombre >= 0.9 and s_valor < 0.2:
                score_total *= 0.5
                just_valor += " => penalizado por valor distinto"

            is_match = score_total >= umbral

            if score_total > best_score:
                best_score = score_total
                best_match = FieldMatch(
                    source_field=src_field,
                    target_field=tgt_field,
                    source_value=src_val,
                    target_value=tgt_val,
                    score_nombre=round(s_nombre, 4),
                    score_valor=round(s_valor, 4),
                    score_total=round(score_total, 4),
                    is_match=is_match,
                    justificacion_nombre=just_nombre,
                    justificacion_valor=just_valor,
                    justificacion_final="",
                )

        if best_match:
            # Determinar justificacion final
            parts = []
            if best_match.score_nombre >= 0.5:
                parts.append("nombre")
            if best_match.score_valor >= 0.5:
                parts.append("valor")

            if not parts:
                best_match.justificacion_final = "sin coincidencia significativa"
            elif len(parts) == 2:
                best_match.justificacion_final = f"match por ambos ({' + '.join(parts)})"
            else:
                best_match.justificacion_final = f"match por {parts[0]}"

            results.append(best_match)

    # Ordenar por score_total descendente
    results.sort(key=lambda m: m.score_total, reverse=True)
    return results


def print_results(matches: list[FieldMatch]) -> str:
    """
    Imprime resultados formateados.
    """
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'SOURCE FIELD':<25} {'TARGET FIELD':<25} {'S_NOM':>6} {'S_VAL':>6} {'TOTAL':>6} {'MATCH':>5} {'JUSTIFICACION'}")
    lines.append("-" * 100)

    for m in matches:
        icon = "MATCH" if m.is_match else "-----"
        lines.append(
            f"{m.source_field:<25} {m.target_field:<25} "
            f"{m.score_nombre:>6.2f} {m.score_valor:>6.2f} "
            f"{m.score_total:>6.2f} {icon:>5} | {m.justificacion_final}"
        )

    lines.append("-" * 100)

    matches_only = [m for m in matches if m.is_match]
    lines.append(f"\nTotal matches: {len(matches_only)}/{len(matches)}")

    for m in matches_only:
        lines.append(f"  {m.source_field} <-> {m.target_field}: {m.justificacion_final}")

    lines.append("=" * 100)
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("DEMO 1: Ejemplo del prompt (RUT + nombre)")
    print("=" * 60)

    source1 = {
        "rut_cliente": "12345678-9",
        "nombre": "juan perez",
    }
    target1 = {
        "client_id": "12345678-9",
        "full_name": "juan perez",
    }

    results1 = match_fields(source1, target1)
    print(print_results(results1))

    print("\n" + "=" * 60)
    print("DEMO 2: Caso empresarial completo")
    print("=" * 60)

    source2 = {
        "customer_id": "C-1001",
        "rut_empresa": "76.123.456-K",
        "nombre_completo": "Maria Gonzalez Silva",
        "correo_electronico": "maria.gonzalez@empresa.cl",
        "telefono_contacto": "+56912345678",
        "fecha_nacimiento": "15/03/1985",
        "direccion": "Av. Providencia 1234, Santiago",
        "monto_total_compras": "1500000",
        "fecha_ultima_compra": "2024-01-15",
        "estado_cliente": "activo",
    }

    target2 = {
        "client_id": "C-1001",
        "tax_id": "76123456-K",
        "full_name": "maria gonzalez silva",
        "email": "maria.gonzalez@empresa.cl",
        "phone_number": "+56 9 1234 5678",
        "birth_date": "15-03-1985",
        "address": "Av. Providencia 1234, Santiago",
        "total_purchases": "1500000",
        "last_purchase_date": "2024-01-15",
        "status": "activo",
        "notas": "cliente preferente",
    }

    results2 = match_fields(source2, target2)
    print(print_results(results2))

    print("\n" + "=" * 60)
    print("DEMO 3: No matches esperados")
    print("=" * 60)

    source3 = {
        "customer_id": "C-1001",
        "first_name": "Juan",
        "email": "juan@email.com",
    }
    target3 = {
        "vendor_id": "V-5000",
        "last_name": "Perez",
        "phone": "+56998765432",
    }

    results3 = match_fields(source3, target3)
    print(print_results(results3))
