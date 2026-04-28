from src.db.config_mongodb import SemanticConfigMongoDb
from src.data_training.gen_datos import generacion_datos

config = SemanticConfigMongoDb()



def verificar_relaciones_criticas():
    """Verificar las relaciones que aparecían con baja puntuación"""
    config = SemanticConfigMongoDb()

    print("\n🔍 VERIFICANDO RELACIONES CRÍTICAS:")
    print("=" * 60)

    relaciones_a_verificar = [
        ("rfc", "registro_federal_contribuyentes"),
        ("curp", "clave_unica_registro_poblacion"),
        ("cp", "codigo_postal"),
        ("sku", "stock_keeping_unit"),
        ("iva", "impuesto_al_valor_agregado"),
        ("KUNNR", "customer_number"),
        ("LIFNR", "vendor_number"),
        ("MATNR", "product_code"),
        ("VBELN", "sales_order_number"),
        ("custbody_mx_cfdi_uuid", "UUID_CFDI_PRINCIPAL"),
        ("custentity_rfc", "tax_id_company"),
        ("cust_email_main", "correo_electronico_principal"),
        ("nombre_completo", "fullname"),
        ("fecha_nacimiento", "dob"),
        ("numero_exterior", "house_number"),
        ("codigo_postal", "zipcode"),
        ("razon_social", "company_legal_name"),
        ("importe_total", "grand_total"),
    ]

    indice = config.get_indice_semantico(refresh_cache=True)

    for a, b in relaciones_a_verificar:
        grupo_a = indice.get(a.lower(), "No encontrado")
        grupo_b = indice.get(b.lower(), "No encontrado")

        if grupo_a == grupo_b and grupo_a != "No encontrado":
            print(f"✅ {a:35} ↔ {b:35} → MISMO GRUPO: {grupo_a}")
        else:
            print(f"❌ {a:35} ↔ {b:35} → DIFERENTE: '{grupo_a}' vs '{grupo_b}'")

    config.close()


def migrar_grupos_semanticos_avanzados(config):
    """Migrar grupos semánticos completos para API/CRM/ERP"""

    grupos_semanticos = {
        # ========== DATOS DE CONTACTO ==========
        "nombre_completo": {
            "fullname", "nombrecompleto", "nombre_completo", "full_name",
            "legalname", "razonsocial", "businessname", "tradingname"
        },

        "nombre_persona": {
            "firstname", "givenname", "primer_nombre", "nombre",
            "first_name", "nombre1", "nombres", "names"
        },

        "apellido_persona": {
            "lastname", "surname", "apellido", "last_name",
            "apellidopaterno", "apellidomaterno", "secondname"
        },

        "email": {
            "email", "correo", "mail", "correoelectronico", "emailaddress",
            "email_addr", "primaryemail", "email_principal", "e_mail"
        },

        "telefono": {
            "phone", "telefono", "tel", "telefonomovil", "mobile",
            "cellphone", "celular", "phonenumber", "phone_number",
            "workphone", "homephone", "fax", "telefono_trabajo"
        },

        # ========== DATOS FISCALES ==========
        "rfc": {
            "rfc", "taxid", "tax_id", "fiscalid", "taxidentifier",
            "registrofederal", "taxnumber", "vatid", "vat_id"
        },

        "curp": {
            "curp", "uniqueid", "personalid", "claveunica"
        },

        "identificacion_fiscal": {
            "cuit", "cuil", "nit", "nif", "cédula", "cedula",
            "passport", "identificacion", "identification"
        },

        # ========== DATOS DE UBICACIÓN ==========
        "direccion_completa": {
            "address", "direccion", "fulladdress", "direccioncompleta",
            "streetaddress", "addressline", "domicilio"
        },

        "calle": {
            "street", "calle", "address1", "addressline1",
            "calleynumero", "streename"
        },

        "numero_exterior": {
            "exterior", "numexterior", "streetnumber", "housenumber",
            "addressnumber", "no_exterior"
        },

        "numero_interior": {
            "interior", "numinterior", "apt", "apartment",
            "suite", "office", "piso", "departamento"
        },

        "colonia": {
            "colonia", "neighborhood", "suburb", "barrio",
            "settlement", "distrito"
        },

        "ciudad": {
            "city", "ciudad", "municipio", "town", "cityname"
        },

        "estado": {
            "state", "estado", "provincia", "province",
            "region", "departamento"
        },

        "pais": {
            "country", "pais", "nation", "countrycode",
            "pais_codigo", "iso_country"
        },

        "codigo_postal": {
            "zip", "zipcode", "postalcode", "codigopostal",
            "cp", "postal_code", "código_postal"
        },

        # ========== DATOS FINANCIEROS ==========
        "monto": {
            "amount", "monto", "importe", "total", "valor",
            "price", "precio", "subtotal", "grandtotal"
        },

        "moneda": {
            "currency", "moneda", "currencycode", "moneda_codigo",
            "currency_iso", "divisa"
        },

        "tasa_impuesto": {
            "taxrate", "impuesto", "tax", "vat", "iva",
            "tax_percentage", "alicuota"
        },

        "descuento": {
            "discount", "descuento", "rebate", "promotion",
            "discountrate", "porcentaje_descuento"
        },

        # ========== DATOS DE FACTURACIÓN ==========
        "folio_fiscal": {
            "uuid", "foliofiscal", "comprobante", "cfdi",
            "invoiceid", "factura", "voucher"
        },

        "folio": {
            "folio", "invoicenumber", "facturanumero", "receiptnumber",
            "documentnumber", "numero_documento"
        },

        "fecha_factura": {
            "invoicedate", "fechafactura", "fechacomprobante",
            "issuedate", "fecha_emision", "fecha_timbrado"
        },

        "fecha_vencimiento": {
            "duedate", "fechavencimiento", "expirationdate",
            "fecha_expiracion", "paymentdue"
        },

        # ========== DATOS DE CLIENTES Y PROVEEDORES ==========
        "cliente": {
            "customer", "cliente", "client", "buyer", "comprador",
            "customerid", "client_id", "customer_number"
        },

        "proveedor": {
            "vendor", "proveedor", "supplier", "seller", "vendedor",
            "provider", "vendorid", "supplier_id"
        },

        "empleado": {
            "employee", "empleado", "staff", "worker", "colaborador",
            "employeeid", "empleado_id"
        },

        # ========== DATOS DE PRODUCTOS ==========
        "producto": {
            "product", "producto", "item", "articulo", "sku",
            "productid", "itemnumber", "partnumber"
        },

        "descripcion_producto": {
            "description", "descripcion", "productdescription",
            "itemdescription", "detalle", "concepto"
        },

        "cantidad": {
            "quantity", "cantidad", "qty", "units", "unidades",
            "amount_units", "numero_unidades"
        },

        "unidad_medida": {
            "unit", "unidad", "measurement", "uom", "unit_of_measure",
            "unidad_medida", "sat_unit"
        },

        # ========== DATOS DE INVENTARIO ==========
        "inventario": {
            "inventory", "stock", "existencia", "almacen",
            "warehouse", "stocklevel", "inventory_qty"
        },

        "almacen": {
            "warehouse", "almacen", "bodega", "location",
            "sucursal", "branch", "warehouse_id"
        },

        # ========== DATOS DE ÓRDENES ==========
        "orden_compra": {
            "purchaseorder", "ordencompra", "po", "po_number",
            "purchase_order", "oc"
        },

        "orden_venta": {
            "salesorder", "ordenventa", "so", "order_number",
            "customerorder", "pedido"
        },

        "status": {
            "status", "estado", "estatus", "state", "condition",
            "situacion", "activo", "activo_inactivo"
        },

        # ========== DATOS TEMPORALES ==========
        "fecha_creacion": {
            "createddate", "fechacreacion", "created_at",
            "fecha_alta", "creationdate", "date_created"
        },

        "fecha_modificacion": {
            "modifieddate", "fechamodificacion", "updated_at",
            "lastmodified", "fecha_actualizacion"
        },

        "fecha_inicio": {
            "startdate", "fechainicio", "start_date",
            "begindate", "fecha_inicial"
        },

        "fecha_fin": {
            "enddate", "fechafin", "end_date", "finisheddate",
            "fecha_final", "expiration_date"
        },

        "año": {
            "year", "año", "ano", "exercise", "fiscalyear",
            "periodo_anual"
        },

        "mes": {
            "month", "mes", "period", "periodo", "monthly"
        },

        # ========== DATOS DE USUARIO Y SISTEMA ==========
        "usuario": {
            "user", "usuario", "username", "userid", "login",
            "email_usuario", "created_by"
        },

        "rol": {
            "role", "rol", "perfil", "profile", "user_role",
            "permission", "nivel_acceso"
        },

        "sesion": {
            "session", "sesion", "token", "accesstoken",
            "apikey", "api_key", "bearer_token"
        },

        # ========== DATOS TÉCNICOS ==========
        "id_sistema": {
            "id", "identifier", "recordid", "entityid", "objectid",
            "primarykey", "pk", "uuid", "guid"
        },

        "codigo": {
            "code", "codigo", "key", "clave", "reference",
            "refcode", "shortcode"
        },

        "nombre_sistema": {
            "name", "nombre", "label", "displayname", "titulo",
            "title", "description_name"
        },

        # ========== DATOS DE TRANSACCIONES ==========
        "tipo_transaccion": {
            "transactiontype", "tipotransaccion", "doctype",
            "documenttype", "tipo_documento", "transaction_code"
        },

        "referencia": {
            "reference", "referencia", "ref", "related",
            "vinculacion", "relacionado"
        },

        "comentario": {
            "comment", "comentario", "note", "nota", "observacion",
            "remarks", "additional_info"
        },

        # ========== DATOS DE CONFIGURACIÓN ==========
        "configuracion": {
            "config", "configuration", "settings", "parametros",
            "parameters", "preferences", "propiedades"
        },

        "version": {
            "version", "versión", "release", "api_version",
            "app_version", "software_version"
        },

        # ========== DATOS DE API ==========
        "endpoint": {
            "endpoint", "url", "path", "route", "api_route",
            "servicio", "resource"
        },

        "metodo_http": {
            "method", "http_method", "verb", "operation",
            "get", "post", "put", "delete", "patch"
        },

        "response": {
            "response", "respuesta", "output", "resultado",
            "payload", "body", "data_response"
        },

        "request": {
            "request", "peticion", "input", "parametros",
            "parameters", "query", "body_request"
        },

        # ========== DATOS DE METADATOS ==========
        "metadata": {
            "metadata", "metadatos", "extra", "additional",
            "customfields", "campos_adicionales", "attributes"
        },

        "origen": {
            "source", "origen", "system", "fuente", "procedencia",
            "integration", "external_source"
        }
    }

    # # Insertar grupos semánticos
    # for grupo, tokens in grupos_semanticos.items():
    #     for token in tokens:
    #         config.add_to_grupo_semantico(grupo, token, idioma="multi")
    #         print(f"Agregado: {token} -> {grupo}")
    #
    # config.refresh_cache()
    # print(f"✅ Migración completada: {len(grupos_semanticos)} grupos semánticos creados")
    #
    # return grupos_semanticos

# training_data = generacion_datos()
# config.guardar_dataset(
#     datos_ent=training_data,
#     nombre="dataset_alpha",
#     version="1.1",
#     metadata={
#         "tipo": "matching_campos",
#         "descripcion": "datasets inicial"
#     }
# )
# config.exportar_dataset_json("69b995ebc6a74270c71fb315","apha")