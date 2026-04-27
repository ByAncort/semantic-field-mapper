"""
datos_entrenamiento.py
~5FalseFalse pares de entrenamiento organizados por categoría.
Cubre: ERP, CRM, SAP, NetSuite, Salesforce, APIs, legacy, cross-idioma.

"""

def fp():
    """
    FALSOS POSITIVOS — label False
    Comparten tokens, prefijos o estructura pero son semánticamente distintos.
    Son los más peligrosos: el modelo tiende a decir True cuando no debe.
    """
    return [

        # ── Mismo campo, diferente entidad ───────────────────────────────────
        # El modelo ve "email" en ambos y dice True. No debe.
        ("customer_email", "vendor_email", False),
        ("customer_email", "employee_email", False),
        ("customer_email", "supplier_email", False),
        ("vendor_phone", "customer_phone", False),
        ("vendor_rfc", "customer_rfc", False),
        ("vendor_address", "customer_address", False),
        ("vendor_name", "customer_name", False),
        ("employee_id", "customer_id", False),
        ("employee_id", "vendor_id", False),
        ("customer_id", "vendor_id", False),
        ("employee_name", "customer_name", False),
        ("employee_email", "vendor_email", False),

        # ── Dirección: mismo nivel ≠ distintos niveles geográficos ───────────
        ("ciudad", "estado", False),
        ("estado", "pais", False),
        ("colonia", "ciudad", False),
        ("colonia", "municipio", False),
        ("municipio", "estado", False),
        ("numero_exterior", "numero_interior", False),
        ("calle", "colonia", False),
        ("codigo_postal", "numero_exterior", False),
        ("city", "state", False),
        ("state", "country", False),
        ("neighborhood", "city", False),
        ("street_number", "postal_code", False),
        ("address_lineTrue", "address_line2", False),
        ("shipping_address", "billing_address", False),
        ("shipping_city", "billing_city", False),
        ("shipping_zip", "billing_zip", False),

        # ── Precio: tipo distinto ─────────────────────────────────────────────
        ("precio_venta", "precio_costo", False),
        ("precio_venta", "precio_compra", False),
        ("sale_price", "cost_price", False),
        ("unit_price", "unit_cost", False),
        ("list_price", "purchase_price", False),
        ("precio_minimo", "precio_maximo", False),
        ("precio_mayoreo", "precio_menudeo", False),

        # ── Stock mín/máx ─────────────────────────────────────────────────────
        ("stock_minimo", "stock_maximo", False),
        ("minimum_stock", "maximum_stock", False),
        ("qty_min", "qty_max", False),

        # ── Fechas: distintas fechas ──────────────────────────────────────────
        ("fecha_creacion", "fecha_modificacion", False),
        ("fecha_inicio", "fecha_fin", False),
        ("fecha_inicio", "fecha_vencimiento", False),
        ("start_date", "end_date", False),
        ("creation_date", "modification_date", False),
        ("due_date", "payment_date", False),
        ("birth_date", "hire_date", False),
        ("fecha_nacimiento", "fecha_contratacion", False),
        ("fecha_emision", "fecha_vencimiento", False),
        ("invoice_date", "due_date", False),
        ("fecha_entrega", "fecha_envio", False),
        ("delivery_date", "ship_date", False),

        # ── Contacto: canal distinto ──────────────────────────────────────────
        ("email", "telefono", False),
        ("email", "celular", False),
        ("email", "whatsapp", False),
        ("telefono", "fax", False),
        ("work_phone", "home_phone", False),
        ("primary_email", "secondary_email", False),
        ("email_principal", "email_alternativo", False),
        ("telefono_casa", "telefono_oficina", False),
        ("celular", "telefono_fijo", False),
        ("phone_mobile", "phone_home", False),

        # ── Nombre: partes distintas del nombre ───────────────────────────────
        ("nombre", "apellido", False),
        ("first_name", "last_name", False),
        ("apellido_paterno", "apellido_materno", False),
        ("nombre", "apellido_paterno", False),
        ("firstname", "lastname", False),
        ("given_name", "surname", False),
        ("nombre_completo", "razon_social", False),  # persona ≠ empresa

        # ── IDs fiscales distintos ────────────────────────────────────────────
        ("rfc", "curp", False),
        ("rfc", "nss", False),
        ("curp", "nss", False),
        ("tax_id", "social_security_number", False),
        ("rfc", "clabe", False),

        # ── UUID ≠ folio (confusión muy común en CFDI) ────────────────────────
        ("cfdi_uuid", "folio_factura", False),
        ("uuid_sat", "serie_factura", False),
        ("invoice_uuid_sat", "invoice_number", False),
        ("custbody_mx_cfdi_uuid", "custrecord_cfdi_serie", False),
        ("folio_fiscal", "folio_interno", False),

        # ── Campos SAT que se confunden ───────────────────────────────────────
        ("metodo_pago", "forma_pago", False),  # PUE/PPD ≠ FalseTrue/False2/False3
        ("uso_cfdi", "regimen_fiscal", False),
        ("uso_cfdi", "metodo_pago", False),
        ("regimen_fiscal", "tipo_persona", False),
        ("cfdi_use_code", "tax_regime_code", False),
        ("payment_method_sat", "payment_form_sat", False),

        # ── Financiero: campos relacionados pero distintos ────────────────────
        ("subtotal", "total", False),
        ("subtotal", "descuento", False),
        ("total", "descuento", False),
        ("iva", "descuento", False),
        ("iva", "isr", False),
        ("tasa_iva", "tasa_isr", False),
        ("net_amount", "gross_amount", False),
        ("tax_amount", "discount_amount", False),
        ("clabe", "cuenta_bancaria", False),
        ("iban", "clabe", False),
        ("swift", "clabe", False),

        # ── Producto: distintos campos del mismo producto ─────────────────────
        ("sku", "upc", False),
        ("sku", "ean", False),
        ("upc", "ean", False),
        ("product_code", "product_name", False),
        ("nombre_producto", "descripcion_producto", False),
        ("codigo_producto", "descripcion_producto", False),
        ("item_code", "item_description", False),

        # ── Transacciones: documentos distintos ───────────────────────────────
        ("folio_pedido", "folio_factura", False),
        ("folio_cotizacion", "folio_pedido", False),
        ("order_id", "invoice_id", False),
        ("quote_number", "order_number", False),
        ("po_number", "invoice_number", False),

        # ── Secuenciales (True vs 2) ─────────────────────────────────────────────
        ("phoneTrue", "phone2", False),
        ("addressTrue", "address2", False),
        ("contact_person_True", "contact_person_2", False),
        ("emailTrue", "email2", False),
        ("referenceTrue", "reference2", False),

        # ── Prefijo cambia semántica ──────────────────────────────────────────
        ("tax_code", "product_code", False),
        ("country_code", "area_code", False),
        ("department_code", "employee_code", False),
        ("customer_email_verified", "customer_email", False),  # verified ≠ email
        ("rfc_temporal", "rfc_definitivo", False),
        ("phone_primary", "phone_secondary", False),
        ("cuenta_principal", "cuenta_secundaria", False),
        ("precio_con_iva", "precio_sin_iva", False),
        ("monto_pagado", "monto_pendiente", False),
        ("saldo_favor", "saldo_cargo", False),
        ("custbody_ship_address", "custbody_bill_address", False),
    ]

def fn():
    """
    FALSOS NEGATIVOS — label True
    Parecen distintos (diferente longitud, idioma, sistema, abreviatura)
    pero representan exactamente el mismo campo.
    Son los más difíciles de aprender porque tienen pocos tokens en común.
    """
    return [

        # ── Siglas SAP vs nombre descriptivo ──────────────────────────────────
        ("KUNNR", "customer_number", True),
        ("KUNNR", "numero_cliente", True),
        ("LIFNR", "vendor_number", True),
        ("LIFNR", "numero_proveedor", True),
        ("MATNR", "material_number", True),
        ("MATNR", "codigo_material", True),
        ("VBELN", "sales_document", True),
        ("VBELN", "numero_orden_venta", True),
        ("BUKRS", "company_code", True),
        ("WERKS", "plant_code", True),
        ("LGORT", "storage_location", True),
        ("MENGE", "quantity", True),
        ("NETWR", "net_value", True),
        ("WAERS", "currency_code", True),

        # ── Siglas Netsuite vs nombre descriptivo ─────────────────────────────
        ("custbody_mx_cfdi_uuid", "uuid_fiscal_sat", True),
        ("custbody_mx_cfdi_uuid", "folio_fiscal_timbre", True),
        ("custentity_mx_rfc", "tax_id_empresa", True),
        ("custbody_mx_uso_cfdi", "cfdi_purpose_code", True),
        ("custbody_mx_metodo_pago", "sat_payment_method", True),
        ("custbody_mx_regimen_fiscal", "fiscal_regime_sat", True),
        ("custcol_mx_uso_cfdi", "column_cfdi_usage", True),

        # ── Salesforce custom fields vs nombre legible ─────────────────────────
        ("CFDI_UUID__c", "uuid_comprobante_fiscal", True),
        ("RFC_Cliente__c", "tax_id_customer", True),
        ("Email_Principal__c", "main_email_address", True),
        ("Razon_Social__c", "company_legal_name", True),
        ("Fecha_Nacimiento__c", "birth_date_contact", True),

        # ── Legacy / bases de datos viejas ────────────────────────────────────
        ("c_nombre", "customer_first_name", True),
        ("c_apellido", "customer_last_name", True),
        ("c_correo", "contact_email_address", True),
        ("t_telefono", "phone_number_contact", True),
        ("f_factura_uuid", "invoice_uuid_sat", True),
        ("fac_uuid_sat", "cfdi_folio_fiscal", True),
        ("cli_id", "customer_identifier", True),
        ("prov_id", "vendor_identifier", True),
        ("cod_prod", "product_code", True),

        # ── Abreviaturas vs completo ───────────────────────────────────────────
        ("rfc", "registro_federal_contribuyentes", True),
        ("curp", "clave_unica_registro_poblacion", True),
        ("cp", "codigo_postal", True),
        ("cp", "postal_code", True),
        ("sku", "stock_keeping_unit", True),
        ("upc", "universal_product_code", True),
        ("iva", "impuesto_valor_agregado", True),
        ("isr", "impuesto_sobre_renta", True),
        ("tel", "numero_telefonico", True),
        ("tel", "phone_number", True),
        ("dir", "domicilio_completo", True),
        ("dir", "full_address", True),
        ("amt", "amount_total", True),
        ("amt", "monto_total", True),
        ("qty", "cantidad_disponible", True),
        ("dob", "fecha_nacimiento", True),
        ("dob", "birth_date", True),
        ("poc", "punto_de_contacto", True),

        # ── Cross-idioma sin tokens compartidos ───────────────────────────────
        ("nombre", "given_name", True),
        ("apellido", "surname", True),
        ("correo", "mail_address", True),
        ("ciudad", "town_name", True),
        ("pais", "nation_code", True),
        ("colonia", "suburb_name", True),
        ("importe", "amount_due", True),
        ("descuento", "rebate_amount", True),
        ("almacen", "warehouse_location", True),
        ("sucursal", "branch_office", True),
        ("empleado", "staff_member", True),
        ("proveedor", "supplier_name", True),
        ("empresa", "organization_name", True),
        ("factura", "invoice_document", True),
        ("pedido", "purchase_order", True),
        ("cotizacion", "sales_quote", True),
        ("existencia", "on_hand_quantity", True),
        ("vigente", "is_active", True),
        ("borrado", "is_deleted", True),

        # ── Formato diferente (camelCase, UPPER, con puntos) ──────────────────
        ("customer_name", "customerName", True),
        ("customer_name", "CUSTOMER_NAME", True),
        ("first_name", "firstName", True),
        ("last_name", "lastName", True),
        ("tax_id", "taxId", True),
        ("tax_id", "TAX_ID", True),
        ("zip_code", "zipCode", True),
        ("zip_code", "ZIP_CODE", True),
        ("email_address", "emailAddress", True),
        ("phone_number", "phoneNumber", True),
        ("invoice_number", "invoiceNumber", True),
        ("billing_address.city", "facturacion.ciudad", True),
        ("customer.contact.email", "cliente.correo", True),
        ("payment.method.code", "pago.metodo", True),

        # ── Con prefijos de sistema que deben ignorarse ───────────────────────
        ("cust_uuid_mx", "folio_fiscal_uuid", True),
        ("sys_customer_id", "id_cliente", True),
        ("erp_vendor_code", "codigo_proveedor", True),
        ("crm_contact_email", "correo_electronico", True),
        ("db_fecha_nacimiento", "birth_date", True),
        ("tbl_product_sku", "codigo_producto", True),
        ("inv_unit_price", "precio_unitario", True),
        ("fin_total_amount", "importe_total", True),
        ("log_ship_date", "fecha_envio", True),
        ("hr_hire_date", "fecha_contratacion", True),

        # ── Singular / plural ─────────────────────────────────────────────────
        ("producto", "productos", True),
        ("cliente", "clientes", True),
        ("proveedor", "proveedores", True),
        ("telefono", "telefonos", True),
        ("email", "emails_contacto", True),
        ("product", "products", True),
        ("customer", "customers", True),
        ("address", "addresses", True),
    ]

def datos_base():
    return [
    ("custbody_mx_cfdi_uuid", "PFR_UUID_TBTrue2False", True),              # UUID de factura
    ("custbody_cfdi_folio_fiscal", "invoice_uuid_sat", True),       # Folio fiscal
    ("custrecord_cfdi_serie", "document_series_code", True),        # Serie de factura
    ("custbody_mx_regimen_fiscal", "tax_regimen_code", True),       # Régimen fiscal
    ("custbody_mx_uso_cfdi", "cfdi_usage_type", True),              # Uso de CFDI
    ("custbody_mx_metodo_pago", "payment_method_sat", True),        # Método de pago

    ("rfc_cliente", "customer_tax_id_mx", True),                    # RFC México
    ("tax_identification_number", "company_rfc", True),             # Tax ID genérico
    ("custentity_rfc", "taxid_number", True),                       # RFC en Netsuite
    ("rfc_proveedor", "vendor_tax_id", True),                       # RFC proveedor
    ("curp_cliente", "customer_curp", True),                        # CURP (México)
    ("rfc", "tax_id", True),                                         # RFC vs Tax ID

    ("customer_name", "nombre_cliente", True),                      # Inglés/Español
    ("firstname", "primer_nombre", True),                           # First name
    ("lastname", "apellido_paterno", True),                         # Last name
    ("mothers_maiden_name", "apellido_materno", True),              # Maternal last name
    ("business_partner_name", "socio_negocio_nombre", True),        # Business partner

    ("email_principal", "primary_email_address", True),             # Email principal
    ("work_email", "correo_trabajo", True),                         # Email trabajo
    ("personal_email", "email_personal", True),                     # Email personal
    ("billing_email", "email_facturacion", True),                   # Email facturación
    ("alternative_email", "correo_alternativo", True),              # Email alternativo

    ("phone_mobile", "celular_contacto", True),                     # Celular
    ("work_phone", "telefono_oficina", True),                       # Teléfono oficina
    ("home_phone", "telefono_casa", True),                          # Teléfono casa
    ("fax_number", "numero_fax", True),                             # Fax
    ("whatsapp_contact", "telefono_whatsapp", True),                # WhatsApp

    ("shipping_address", "direccion_envio", True),  # Dirección envío
    ("billing_address", "direccion_facturacion", True),  # Dirección facturación
    ("street_name", "calle_direccion", True),  # Calle
    ("street_number", "numero_exterior", True),  # Número exterior
    ("interior_number", "numero_interior", True),  # Número interior
    ("neighborhood", "colonia_ubicacion", True),  # Colonia
    ("city", "ciudad_municipio", True),  # Ciudad
    ("state_province", "estado_region", True),  # Estado
    ("postal_code", "codigo_postal_cp", True),  # Código postal
    ("country_code", "pais_iso", True),  # País

    ("full_address", "direccion_completa", True),                   # Dirección completa
    ("shipping_full_address", "direccion_envio_completa", True),    # Envío completa
    ("billing_full_address", "direccion_fact_completa", True),      # Facturación completa

    ("account_number", "numero_cuenta_contable", True),             # Número cuenta
    ("bank_account", "cuenta_bancaria", True),                      # Cuenta bancaria
    ("clabe_interbancaria", "bank_clabe", True),                    # CLABE (México)
    ("credit_card", "tarjeta_credito", True),                       # Tarjeta crédito
    ("payment_method", "metodo_pago", True),                        # Método pago
    ("payment_terms", "condiciones_pago", True),                    # Términos pago

    ("sku_code", "codigo_producto", True),                          # SKU
    ("upc_code", "codigo_barras", True),                            # UPC
    ("product_name", "nombre_producto", True),                      # Nombre producto
    ("product_description", "descripcion_producto", True),          # Descripción
    ("unit_price", "precio_unitario", True),                        # Precio unitario
    ("sale_price", "precio_venta", True),                           # Precio venta
    ("cost_price", "precio_costo", True),                           # Precio costo
    ("quantity_in_stock", "inventario_disponible", True),           # Cantidad inventario
    ("minimum_stock", "stock_minimo", True),                        # Stock mínimo
    ("maximum_stock", "stock_maximo", True),                        # Stock máximo

    ("creation_date", "fecha_creacion", True),                      # Fecha creación
    ("modification_date", "fecha_modificacion", True),              # Fecha modificación
    ("due_date", "fecha_vencimiento", True),                        # Fecha vencimiento
    ("birth_date", "fecha_nacimiento", True),                       # Fecha nacimiento
    ("start_date", "fecha_inicio", True),                           # Fecha inicio
    ("end_date", "fecha_fin", True),                                # Fecha fin
    ("transaction_date", "fecha_transaccion", True),                # Fecha transacción

    ("order_id", "numero_pedido", True),                            # Número pedido
    ("invoice_number", "folio_factura", True),                      # Folio factura
    ("quote_number", "numero_cotizacion", True),                    # Número cotización
    ("po_number", "numero_orden_compra", True),                     # Orden compra
    ("customer_id", "id_cliente", True),                            # ID cliente
    ("vendor_id", "id_proveedor", True),                            # ID proveedor
    ("employee_id", "id_empleado", True),                           # ID empleado

    ("is_active", "activo_flag", True),                             # Activo/Inactivo
    ("is_deleted", "eliminado_flag", True),                         # Eliminado
    ("is_taxable", "aplica_impuesto", True),                        # Aplica impuesto
    ("is_shippable", "requiere_envio", True),                       # Requiere envío
    ("is_discountable", "aplica_descuento", True),                  # Aplica descuento

        ("custbody_mx_cfdi_uuid", "CFDI_UUID_PRINCIPAL", True),  # Abreviatura vs completo
        ("tel", "telefono_contacto", True),  # Tel vs teléfono
        ("fono", "numero_telefonico", True),  # Fono vs teléfono
        ("dir", "direccion_comercial", True),  # Dir vs dirección
        ("ref", "referencia_cliente", True),  # Ref vs referencia
        ("desc", "descripcion_larga", True),  # Desc vs descripción
        ("qty", "cantidad_producto", True),  # Qty vs cantidad
        ("amt", "monto_total", True),  # Amt vs monto
        ("txn", "transaccion_id", True),  # Txn vs transacción

        # Diferentes prefijos/sufijos
        ("customer_email", "cliente_correo", True),  # Diferente orden
        ("email_cliente", "customer_mail", True),  # Intercambio
        ("client_email_address", "correo_electronico_cliente", True),  # Largo vs corto
        ("tax_id_number", "rfc_company", True),  # Tax ID vs RFC
        ("company_tax_id", "rfc_empresa", True),  # Company vs empresa

        # Sin prefijos/sufijos comunes
        ("uuid", "folio_fiscal_uuid", True),  # UUID genérico
        ("rfc", "registro_federal_contribuyentes", True),  # Sigla vs completo
        ("curp", "clave_unica_registro_poblacion", True),  # Sigla vs completo
        ("sku", "stock_keeping_unit", True),  # Sigla vs completo
        ("upc", "universal_product_code", True),  # Sigla vs completo

        # Con guiones bajos y sin ellos
        ("customer_name", "customername", True),  # Con/sin guión bajo
        ("first_name", "firstname", True),  # Con/sin guión bajo
        ("last_name", "lastname", True),  # Con/sin guión bajo
        ("tax_id", "taxid", True),  # Con/sin guión bajo
        ("zip_code", "zipcode", True),  # Con/sin guión bajo

        # Singular/Plural
        ("customer", "clientes", True),  # Singular/Plural
        ("product", "productos", True),  # Singular/Plural
        ("email", "emails_contacto", True),  # Singular/Plural
        ("phone", "telefonos", True),  # Singular/Plural
        # ============================================
        # 3. CASOS NEGATIVOS (DEBERÍAN SER FALSE)
        # ============================================

        # Confusiones comunes
        ("email", "website_url", False),  # Email vs sitio web
        ("phone", "fax_number", False),  # Teléfono vs fax
        ("rfc", "curp", False),  # RFC vs CURP (diferentes)
        ("sku", "upc_code", False),  # SKU vs UPC
        ("first_name", "last_name", False),  # Nombre vs apellido
        ("nombre", "apellido", False),  # Nombre vs apellido
        ("price", "quantity", False),  # Precio vs cantidad
        ("discount", "tax", False),  # Descuento vs impuesto
        ("subtotal", "total", False),  # Subtotal vs total
        ("credit_card", "bank_account", False),  # Tarjeta vs cuenta

        # Campos completamente diferentes
        ("customer_name", "invoice_date", False),  # Nombre vs fecha
        ("email", "postal_code", False),  # Email vs código postal
        ("phone", "product_sku", False),  # Teléfono vs SKU
        ("address", "payment_method", False),  # Dirección vs pago
        ("rfc", "product_name", False),  # RFC vs producto
        ("uuid", "unit_price", False),  # UUID vs precio
        ("sku", "customer_email", False),  # SKU vs email
        ("quantity", "discount_rate", False),  # Cantidad vs descuento

        # Prefijos engañosos pero diferentes
        ("customer_email", "employee_email", False),  # Cliente vs empleado
        ("vendor_rfc", "customer_rfc", False),  # Proveedor vs cliente
        ("shipping_address", "billing_address", False),  # Envío vs facturación
        ("work_phone", "home_phone", False),  # Trabajo vs casa
        ("primary_email", "secondary_email", False),  # Principal vs secundario

        # Fechas diferentes
        ("start_date", "end_date", False),  # Inicio vs fin
        ("creation_date", "modification_date", False),  # Creación vs modificación
        ("due_date", "payment_date", False),  # Vencimiento vs pago
        ("birth_date", "hire_date", False),  # Nacimiento vs contratación

        # Números de identificación diferentes
        ("account_number", "customer_id", False),  # Cuenta vs ID cliente
        ("order_id", "invoice_id", False),  # Pedido vs factura
        ("quote_number", "po_number", False),  # Cotización vs orden compra
        ("employee_id", "vendor_id", False),  # Empleado vs proveedor

        # ============================================
        # 4. CASOS TRAMPA (EDGE CASES)
        # ============================================

        # Muy similares pero diferentes
        ("customer_email", "customer_email_verified", False),  # Email vs email verificado
        ("rfc_temporal", "rfc_definitivo", False),  # Temporal vs definitivo
        ("phone_primary", "phone_secondary", False),  # Primario vs secundario
        ("address_lineTrue", "address_line2", False),  # Línea True vs línea 2

        # Con números pero diferentes
        ("phoneTrue", "phone2", False),  # Teléfono True vs 2
        ("address_True", "address_2", False),  # Dirección True vs 2
        ("contact_person_True", "contact_person_2", False),  # Contacto True vs 2

        # Palabras comunes en diferente contexto
        ("product_code", "tax_code", False),  # Código producto vs impuesto
        ("country_code", "area_code", False),  # País vs área
        ("department_code", "employee_code", False),  # Depto vs empleado

        # ============================================
        # 5. CASOS DE PRODUCCIÓN REALES
        # ============================================

        # Netsuite (ERP)
        ("custbody_mx_cfdi_uuid", "custrecord_pfr_invoice_uuid", True),  # Netsuite fields
        ("custentity_mx_rfc", "custrecord_tax_id", True),  # Entity vs record
        ("custcol_mx_uso_cfdi", "custrecord_cfdi_usage", True),  # Column vs record

        # Salesforce
        ("CFDI_UUID__c", "Invoice_UUID__c", True),  # Salesforce custom fields
        ("RFC_Cliente__c", "Tax_ID__c", True),  # Salesforce
        ("Email_Principal__c", "Primary_Email__c", True),  # Salesforce

        # SAP
        ("KUNNR", "Customer_Number", True),  # SAP customer
        ("LIFNR", "Vendor_Number", True),  # SAP vendor
        ("MATNR", "Material_Number", True),  # SAP material
        ("VBELN", "Sales_Document", True),  # SAP sales doc

        # Bases de datos legacy
        ("c_nombre", "cli_firstname", True),  # Legacy prefixes
        ("t_telefono", "tel_contacto", True),  # Legacy tables
        ("f_factura_uuid", "fac_uuid_sat", True),  # Legacy factura

        # APIs externas
        ("billing_address.city", "facturacion.direccion.ciudad", True),  # API nested
        ("customer.contact.email", "cliente.contacto.correo", True),  # API nested
        ("payment.method.code", "pago.metodo.codigo", True),  # API nested

        # Mezcla de idiomas
        ("customer_rfc", "cliente_tax_id", True),  # Eng/Spa mix
        ("nombre_producto", "product_name", True),  # Spa/Eng mix
        ("precio_venta", "sale_amount", True),  # Spa/Eng mix

        ("customer_email","vendor_email",False),
        ("shipping_address","billing_address",False),
        ("precio_venta","precio_costo",False),
        ("stock_minimo","stock_maximo",False),
        ("id_cliente","id_proveedor",False),
        ("work_phone","home_phone",False),
    ]

def similares():
    """Pares que SÍ son el mismo campo — label True"""
    return [

        # ══════════════════════════════════════════════════════
        # UUID / CFDI / FISCAL MEXICANO
        # ══════════════════════════════════════════════════════
        ("uuid",                            "folio_fiscal",                     True),
        ("uuid",                            "timbre_uuid",                      True),
        ("uuid",                            "uuid_sat",                         True),
        ("cfdi_uuid",                       "uuid_timbrado",                    True),
        ("cfdi_uuid",                       "folio_fiscal_sat",                 True),
        ("folio_fiscal",                    "timbre_fiscal_digital",            True),
        ("custbody_mx_cfdi_uuid",           "uuid_comprobante",                 True),
        ("custbody_mx_cfdi_uuid",           "cfdi_folio_fiscal",                True),
        ("CFDI_UUID__c",                    "folio_fiscal_uuid",                True),
        ("Invoice_UUID__c",                 "uuid_sat",                         True),
        ("f_factura_uuid",                  "cfdi_uuid",                        True),
        ("fac_uuid_sat",                    "uuid_timbrado",                    True),
        ("uuid_cfdi_principal",             "folio_fiscal_timbre",              True),
        ("serie_cfdi",                      "document_series_code",             True),
        ("serie_factura",                   "invoice_series",                   True),
        ("regimen_fiscal",                  "fiscal_regime",                    True),
        ("regimen_fiscal",                  "tax_regime_code",                  True),
        ("custbody_mx_regimen_fiscal",      "regime_sat_code",                  True),
        ("uso_cfdi",                        "cfdi_use",                         True),
        ("uso_cfdi",                        "cfdi_purpose",                     True),
        ("custbody_mx_uso_cfdi",            "cfdi_usage_type",                  True),
        ("metodo_pago",                     "payment_method",                   True),
        ("metodo_pago",                     "sat_payment_method",               True),
        ("custbody_mx_metodo_pago",         "payment_method_code",              True),
        ("forma_pago",                      "payment_form",                     True),
        ("forma_pago",                      "way_of_payment",                   True),
        ("condicion_pago",                  "payment_terms",                    True),
        ("condiciones_pago",                "terms_of_payment",                 True),
        ("tipo_comprobante",                "invoice_type",                     True),
        ("tipo_comprobante",                "cfdi_type",                        True),
        ("certificado_sat",                 "sat_certificate",                  True),
        ("no_certificado",                  "certificate_number",               True),
        ("sello_digital",                   "digital_stamp",                    True),
        ("sello_sat",                       "sat_seal",                         True),

        # ══════════════════════════════════════════════════════
        # RFC / CURP / IDs FISCALES
        # ══════════════════════════════════════════════════════
        ("rfc",                             "tax_id",                           True),
        ("rfc",                             "taxpayer_id",                      True),
        ("rfc",                             "fiscal_id",                        True),
        ("rfc",                             "id_fiscal",                        True),
        ("rfc",                             "registro_federal_contribuyentes",   True),
        ("rfc_cliente",                     "customer_tax_id",                  True),
        ("rfc_empresa",                     "company_tax_id",                   True),
        ("rfc_proveedor",                   "vendor_tax_id",                    True),
        ("custentity_rfc",                  "taxid_number",                     True),
        ("custentity_mx_rfc",               "tax_id_company",                   True),
        ("RFC_Cliente__c",                  "Tax_ID__c",                        True),
        ("tax_identification_number",       "company_rfc",                      True),
        ("curp",                            "clave_unica_registro_poblacion",    True),
        ("curp",                            "personal_id_mx",                   True),
        ("curp_cliente",                    "customer_curp",                    True),
        ("curp_empleado",                   "employee_curp_mx",                 True),
        ("nss",                             "numero_seguro_social",             True),
        ("nss",                             "social_security_number_mx",        True),
        ("nss_empleado",                    "employee_social_security",         True),

        # ══════════════════════════════════════════════════════
        # NOMBRE DE PERSONA
        # ══════════════════════════════════════════════════════
        ("nombre",                          "first_name",                       True),
        ("nombre",                          "given_name",                       True),
        ("nombre",                          "primer_nombre",                    True),
        ("nombre",                          "forename",                         True),
        ("first_name",                      "firstname",                        True),
        ("first_name",                      "nombre_propio",                    True),
        ("c_nombre",                        "customer_first_name",              True),
        ("cli_firstname",                   "nombre_cliente",                   True),
        ("apellido",                        "last_name",                        True),
        ("apellido",                        "surname",                          True),
        ("apellido_paterno",                "fathers_surname",                  True),
        ("apellido_paterno",                "first_surname",                    True),
        ("apellido_materno",                "mothers_surname",                  True),
        ("apellido_materno",                "second_surname",                   True),
        ("nombre_completo",                 "full_name",                        True),
        ("nombre_completo",                 "fullname",                         True),
        ("nombre_completo",                 "complete_name",                    True),
        ("nombre_completo",                 "display_name",                     True),
        ("CUSTOMER_NAME",                   "nombre_cliente",                   True),
        ("CUSTOMER_NAME",                   "customer_fullname",                True),
        ("razon_social",                    "business_name",                    True),
        ("razon_social",                    "company_name",                     True),
        ("razon_social",                    "company_legal_name",               True),
        ("razon_social",                    "legal_name",                       True),
        ("razon_social",                    "organization_name",                True),
        ("Razon_Social__c",                 "company_legal_name",               True),
        ("nombre_comercial",                "trade_name",                       True),
        ("nombre_comercial",                "dba_name",                         True),

        # ══════════════════════════════════════════════════════
        # EMAIL
        # ══════════════════════════════════════════════════════
        ("email",                           "correo",                           True),
        ("email",                           "correo_electronico",               True),
        ("email",                           "mail",                             True),
        ("email",                           "email_address",                    True),
        ("email",                           "electronic_mail",                  True),
        ("email_principal",                 "primary_email",                    True),
        ("email_principal",                 "main_email",                       True),
        ("Email_Principal__c",              "primary_email_address",            True),
        ("correo_electronico",              "email_contact",                    True),
        ("customer_email",                  "correo_cliente",                   True),
        ("customer.contact.email",          "cliente.correo",                   True),
        ("crm_contact_email",               "correo_electronico",               True),

        # ══════════════════════════════════════════════════════
        # TELÉFONO
        # ══════════════════════════════════════════════════════
        ("telefono",                        "phone",                            True),
        ("telefono",                        "phone_number",                     True),
        ("telefono",                        "telephone",                        True),
        ("telefono",                        "tel",                              True),
        ("telefono",                        "fono",                             True),
        ("celular",                         "mobile",                           True),
        ("celular",                         "mobile_phone",                     True),
        ("celular",                         "cell_phone",                       True),
        ("celular",                         "movil",                            True),
        ("t_telefono",                      "phone_number_contact",             True),
        ("tel_contacto",                    "contact_phone",                    True),

        # ══════════════════════════════════════════════════════
        # DIRECCIÓN
        # ══════════════════════════════════════════════════════
        ("calle",                           "street",                           True),
        ("calle",                           "street_name",                      True),
        ("numero_exterior",                 "street_number",                    True),
        ("numero_exterior",                 "house_number",                     True),
        ("numero_interior",                 "interior_number",                  True),
        ("numero_interior",                 "suite_number",                     True),
        ("numero_interior",                 "apt_number",                       True),
        ("colonia",                         "neighborhood",                     True),
        ("colonia",                         "suburb",                           True),
        ("municipio",                       "municipality",                     True),
        ("ciudad",                          "city",                             True),
        ("ciudad",                          "town",                             True),
        ("estado",                          "state",                            True),
        ("estado",                          "province",                         True),
        ("pais",                            "country",                          True),
        ("pais",                            "country_code",                     True),
        ("codigo_postal",                   "postal_code",                      True),
        ("codigo_postal",                   "zip_code",                         True),
        ("codigo_postal",                   "zipcode",                          True),
        ("codigo_postal",                   "zip",                              True),
        ("cp",                              "codigo_postal",                    True),
        ("cp",                              "postal_code",                      True),
        ("direccion_completa",              "full_address",                     True),
        ("domicilio_completo",              "complete_address",                 True),
        ("billing_address.city",            "facturacion.direccion.ciudad",     True),

        # ══════════════════════════════════════════════════════
        # PRODUCTO / INVENTARIO
        # ══════════════════════════════════════════════════════
        ("sku",                             "codigo_producto",                  True),
        ("sku",                             "product_code",                     True),
        ("sku",                             "item_code",                        True),
        ("sku",                             "stock_keeping_unit",               True),
        ("sku",                             "part_number",                      True),
        ("sku_code",                        "cod_producto",                     True),
        ("upc",                             "universal_product_code",           True),
        ("upc",                             "barcode",                          True),
        ("ean",                             "european_article_number",          True),
        ("MATNR",                           "material_number",                  True),
        ("MATNR",                           "codigo_material",                  True),
        ("MATNR",                           "product_code",                     True),
        ("nombre_producto",                 "product_name",                     True),
        ("nombre_producto",                 "item_name",                        True),
        ("nombre_articulo",                 "article_name",                     True),
        ("descripcion_producto",            "product_description",              True),
        ("descripcion_producto",            "item_description",                 True),
        ("existencia",                      "stock",                            True),
        ("existencia",                      "quantity_in_stock",                True),
        ("existencia",                      "on_hand_qty",                      True),
        ("inventario_disponible",           "available_quantity",               True),
        ("unidad_medida",                   "unit_of_measure",                  True),
        ("unidad_medida",                   "uom",                              True),
        ("categoria_producto",              "product_category",                 True),
        ("familia_producto",                "product_family",                   True),
        ("almacen",                         "warehouse",                        True),
        ("almacen",                         "warehouse_location",               True),
        ("ubicacion_almacen",               "bin_location",                     True),

        # ══════════════════════════════════════════════════════
        # PRECIOS
        # ══════════════════════════════════════════════════════
        ("precio_unitario",                 "unit_price",                       True),
        ("precio_unitario",                 "price_per_unit",                   True),
        ("precio_lista",                    "list_price",                       True),
        ("precio_lista",                    "catalog_price",                    True),
        ("importe",                         "amount",                           True),
        ("importe",                         "amount_due",                       True),
        ("monto_total",                     "total_amount",                     True),
        ("importe_total",                   "grand_total",                      True),
        ("monto_neto",                      "net_amount",                       True),
        ("subtotal",                        "sub_total",                        True),
        ("subtotal",                        "net_amount",                       True),
        ("total",                           "total_amount",                     True),
        ("total",                           "grand_total",                      True),
        ("descuento",                       "discount",                         True),
        ("descuento",                       "discount_amount",                  True),
        ("porcentaje_descuento",            "discount_rate",                    True),
        ("iva",                             "vat",                              True),
        ("iva",                             "tax_amount",                       True),
        ("iva",                             "sales_tax",                        True),
        ("tasa_iva",                        "vat_rate",                         True),
        ("monto_iva",                       "tax_amount",                       True),
        ("tipo_cambio",                     "exchange_rate",                    True),
        ("moneda",                          "currency",                         True),
        ("moneda",                          "currency_code",                    True),

        # ══════════════════════════════════════════════════════
        # FECHAS
        # ══════════════════════════════════════════════════════
        ("fecha_nacimiento",                "birth_date",                       True),
        ("fecha_nacimiento",                "birthdate",                        True),
        ("fecha_nacimiento",                "date_of_birth",                    True),
        ("fecha_nacimiento",                "dob",                              True),
        ("fecha_creacion",                  "creation_date",                    True),
        ("fecha_creacion",                  "created_at",                       True),
        ("fecha_creacion",                  "date_created",                     True),
        ("fecha_modificacion",              "modification_date",                True),
        ("fecha_modificacion",              "updated_at",                       True),
        ("fecha_modificacion",              "last_modified",                    True),
        ("fecha_vencimiento",               "due_date",                         True),
        ("fecha_vencimiento",               "expiry_date",                      True),
        ("fecha_vencimiento",               "expiration_date",                  True),
        ("fecha_emision",                   "issue_date",                       True),
        ("fecha_emision",                   "emission_date",                    True),
        ("fecha_emision",                   "invoice_date",                     True),
        ("fecha_contratacion",              "hire_date",                        True),
        ("fecha_contratacion",              "employment_date",                  True),
        ("fecha_envio",                     "ship_date",                        True),
        ("fecha_entrega",                   "delivery_date",                    True),
        ("fecha_pago",                      "payment_date",                     True),
        ("fecha_pago",                      "date_of_payment",                  True),
        ("db_fecha_nacimiento",             "birth_date",                       True),
        ("log_ship_date",                   "fecha_envio",                      True),

        # ══════════════════════════════════════════════════════
        # IDs DE TRANSACCIONES
        # ══════════════════════════════════════════════════════
        ("folio_factura",                   "invoice_number",                   True),
        ("folio_factura",                   "invoice_id",                       True),
        ("folio_factura",                   "bill_number",                      True),
        ("numero_factura",                  "invoice_number",                   True),
        ("VBELN",                           "sales_document",                   True),
        ("VBELN",                           "sales_order_number",               True),
        ("VBELN",                           "numero_orden_venta",               True),
        ("folio_pedido",                    "order_number",                     True),
        ("folio_pedido",                    "order_id",                         True),
        ("numero_pedido",                   "purchase_order_number",            True),
        ("po_number",                       "numero_orden_compra",              True),
        ("folio_cotizacion",                "quote_number",                     True),
        ("folio_cotizacion",                "quotation_id",                     True),
        ("numero_cotizacion",               "quote_reference",                  True),
        ("folio_remision",                  "delivery_note_number",             True),
        ("id_cliente",                      "customer_id",                      True),
        ("id_cliente",                      "client_id",                        True),
        ("codigo_cliente",                  "customer_code",                    True),
        ("KUNNR",                           "customer_number",                  True),
        ("KUNNR",                           "numero_cliente",                   True),
        ("KUNNR",                           "id_cliente",                       True),
        ("cli_id",                          "customer_identifier",              True),
        ("id_proveedor",                    "vendor_id",                        True),
        ("id_proveedor",                    "supplier_id",                      True),
        ("codigo_proveedor",                "vendor_code",                      True),
        ("LIFNR",                           "vendor_number",                    True),
        ("LIFNR",                           "numero_proveedor",                 True),
        ("prov_id",                         "vendor_identifier",                True),
        ("id_empleado",                     "employee_id",                      True),
        ("numero_empleado",                 "employee_number",                  True),

        # ══════════════════════════════════════════════════════
        # BANCARIO
        # ══════════════════════════════════════════════════════
        ("cuenta_bancaria",                 "bank_account",                     True),
        ("cuenta_bancaria",                 "bank_account_number",              True),
        ("numero_cuenta",                   "account_number",                   True),
        ("clabe",                           "clabe_interbancaria",              True),
        ("clabe",                           "bank_clabe",                       True),
        ("clabe",                           "interbank_clabe",                  True),
        ("iban",                            "international_bank_account",       True),
        ("swift",                           "swift_code",                       True),
        ("swift",                           "bic_code",                         True),
        ("banco",                           "bank_name",                        True),
        ("nombre_banco",                    "bank_name",                        True),

        # ══════════════════════════════════════════════════════
        # SAP CODES
        # ══════════════════════════════════════════════════════
        ("BUKRS",                           "company_code",                     True),
        ("BUKRS",                           "codigo_empresa",                   True),
        ("WERKS",                           "plant_code",                       True),
        ("WERKS",                           "codigo_planta",                    True),
        ("LGORT",                           "storage_location",                 True),
        ("LGORT",                           "ubicacion_almacen",                True),
        ("MENGE",                           "quantity",                         True),
        ("MENGE",                           "cantidad",                         True),
        ("NETWR",                           "net_value",                        True),
        ("NETWR",                           "valor_neto",                       True),
        ("WAERS",                           "currency_code",                    True),
        ("WAERS",                           "moneda",                           True),
        ("EKGRP",                           "purchasing_group",                 True),
        ("KTOKK",                           "account_group",                    True),

        # ══════════════════════════════════════════════════════
        # ABREVIATURAS → NOMBRE COMPLETO
        # ══════════════════════════════════════════════════════
        ("dir",                             "direccion",                        True),
        ("dir",                             "domicilio",                        True),
        ("tel",                             "telefono",                         True),
        ("ref",                             "referencia",                       True),
        ("desc",                            "descripcion",                      True),
        ("qty",                             "cantidad",                         True),
        ("qty",                             "quantity",                         True),
        ("amt",                             "monto",                            True),
        ("amt",                             "importe",                          True),
        ("dob",                             "fecha_nacimiento",                 True),
        ("poc",                             "punto_de_contacto",                True),
        ("uom",                             "unidad_medida",                    True),
        ("num",                             "numero",                           True),
        ("cod",                             "codigo",                           True),
        ("id",                              "identificador",                    True),

        # ══════════════════════════════════════════════════════
        # FORMATO DIFERENTE (camelCase, UPPER, sin guiones)
        # ══════════════════════════════════════════════════════
        ("customer_name",                   "customerName",                     True),
        ("customer_name",                   "CUSTOMER_NAME",                    True),
        ("first_name",                      "firstName",                        True),
        ("last_name",                       "lastName",                         True),
        ("tax_id",                          "taxId",                            True),
        ("tax_id",                          "TAX_ID",                           True),
        ("zip_code",                        "zipCode",                          True),
        ("email_address",                   "emailAddress",                     True),
        ("phone_number",                    "phoneNumber",                      True),
        ("invoice_number",                  "invoiceNumber",                    True),
        ("birth_date",                      "birthDate",                        True),
        ("creation_date",                   "creationDate",                     True),
        ("customer_id",                     "customerId",                       True),
        ("vendor_id",                       "vendorId",                         True),
        ("product_code",                    "productCode",                      True),

        # ══════════════════════════════════════════════════════
        # PREFIJOS DE SISTEMA QUE DEBEN IGNORARSE
        # ══════════════════════════════════════════════════════
        ("sys_customer_id",                 "id_cliente",                       True),
        ("erp_vendor_code",                 "codigo_proveedor",                 True),
        ("crm_contact_email",               "correo_electronico",               True),
        ("tbl_product_sku",                 "codigo_producto",                  True),
        ("inv_unit_price",                  "precio_unitario",                  True),
        ("fin_total_amount",                "importe_total",                    True),
        ("hr_hire_date",                    "fecha_contratacion",               True),
        ("db_birth_date",                   "fecha_nacimiento",                 True),

        # ══════════════════════════════════════════════════════
        # SINGULAR / PLURAL
        # ══════════════════════════════════════════════════════
        ("producto",                        "productos",                        True),
        ("cliente",                         "clientes",                         True),
        ("proveedor",                       "proveedores",                      True),
        ("telefono",                        "telefonos",                        True),
        ("product",                         "products",                         True),
        ("customer",                        "customers",                        True),
        ("address",                         "addresses",                        True),
        ("invoice",                         "invoices",                         True),

        # ══════════════════════════════════════════════════════
        # LEGACY DATABASES
        # ══════════════════════════════════════════════════════
        ("c_nombre",                        "cli_firstname",                    True),
        ("c_apellido",                      "cli_lastname",                     True),
        ("c_correo",                        "contact_email",                    True),
        ("t_telefono",                      "tel_contacto",                     True),
        ("f_factura_uuid",                  "invoice_uuid_sat",                 True),
        ("fac_uuid_sat",                    "cfdi_folio_fiscal",                True),
        ("cod_prod",                        "product_code",                     True),
        ("cli_id",                          "customer_id",                      True),
        ("prov_id",                         "vendor_id",                        True),

        # ══════════════════════════════════════════════════════
        # FLAGS / BOOLEANOS
        # ══════════════════════════════════════════════════════
        ("activo",                          "is_active",                        True),
        ("activo",                          "active",                           True),
        ("vigente",                         "is_active",                        True),
        ("vigente",                         "enabled",                          True),
        ("eliminado",                       "is_deleted",                       True),
        ("eliminado",                       "deleted",                          True),
        ("borrado",                         "soft_deleted",                     True),
        ("aplica_iva",                      "is_taxable",                       True),
        ("requiere_envio",                  "is_shippable",                     True),
        ("aplica_descuento",                "is_discountable",                  True),
    ]

def distintos():
    """Pares que NO son el mismo campo — label False"""
    return [

        # ══════════════════════════════════════════════════════
        # MISMA ENTIDAD BASE, DIFERENTE TITULAR
        # ══════════════════════════════════════════════════════
        ("customer_email",                  "vendor_email",                     False),
        ("customer_email",                  "employee_email",                   False),
        ("customer_email",                  "supplier_email",                   False),
        ("customer_phone",                  "vendor_phone",                     False),
        ("customer_phone",                  "employee_phone",                   False),
        ("customer_name",                   "vendor_name",                      False),
        ("customer_name",                   "employee_name",                    False),
        ("customer_address",                "vendor_address",                   False),
        ("customer_id",                     "vendor_id",                        False),
        ("customer_id",                     "employee_id",                      False),
        ("vendor_id",                       "employee_id",                      False),
        ("vendor_rfc",                      "customer_rfc",                     False),
        ("customer_rfc",                    "employee_rfc",                     False),
        ("vendor_name",                     "employee_name",                    False),
        ("customer_email",                  "vendor_email",                     False),
        ("client_id",                       "supplier_id",                      False),
        ("customer_code",                   "vendor_code",                      False),
        ("customer_tax_id",                 "employee_tax_id",                  False),

        # ══════════════════════════════════════════════════════
        # DIRECCIÓN: TIPO DISTINTO
        # ══════════════════════════════════════════════════════
        ("shipping_address",                "billing_address",                  False),
        ("shipping_city",                   "billing_city",                     False),
        ("shipping_zip",                    "billing_zip",                      False),
        ("shipping_country",                "billing_country",                  False),
        ("shipping_state",                  "billing_state",                    False),
        ("domicilio_envio",                 "domicilio_fiscal",                 False),
        ("direccion_entrega",               "direccion_facturacion",            False),
        ("custbody_ship_address",           "custbody_bill_address",            False),

        # ══════════════════════════════════════════════════════
        # NIVELES GEOGRÁFICOS DISTINTOS
        # ══════════════════════════════════════════════════════
        ("ciudad",                          "estado",                           False),
        ("estado",                          "pais",                             False),
        ("colonia",                         "ciudad",                           False),
        ("colonia",                         "municipio",                        False),
        ("municipio",                       "estado",                           False),
        ("numero_exterior",                 "numero_interior",                  False),
        ("calle",                           "colonia",                          False),
        ("codigo_postal",                   "numero_exterior",                  False),
        ("city",                            "state",                            False),
        ("state",                           "country",                          False),
        ("neighborhood",                    "city",                             False),
        ("address_lineTrue",                   "address_line2",                    False),

        # ══════════════════════════════════════════════════════
        # NOMBRE: PARTES DISTINTAS
        # ══════════════════════════════════════════════════════
        ("nombre",                          "apellido",                         False),
        ("nombre",                          "apellido_paterno",                 False),
        ("nombre",                          "apellido_materno",                 False),
        ("first_name",                      "last_name",                        False),
        ("first_name",                      "middle_name",                      False),
        ("apellido_paterno",                "apellido_materno",                 False),
        ("given_name",                      "surname",                          False),
        ("nombre_completo",                 "razon_social",                     False),  # persona ≠ empresa

        # ══════════════════════════════════════════════════════
        # EMAIL: TIPO DISTINTO
        # ══════════════════════════════════════════════════════
        ("email_principal",                 "email_alternativo",                False),
        ("email_principal",                 "email_trabajo",                    False),
        ("primary_email",                   "secondary_email",                  False),
        ("work_email",                      "personal_email",                   False),
        ("work_email",                      "home_email",                       False),
        ("billing_email",                   "shipping_email",                   False),
        ("email",                           "email_verified",                   False),
        ("correo_principal",                "correo_alternativo",               False),

        # ══════════════════════════════════════════════════════
        # TELÉFONO: TIPO DISTINTO
        # ══════════════════════════════════════════════════════
        ("telefono",                        "fax",                              False),
        ("work_phone",                      "home_phone",                       False),
        ("work_phone",                      "mobile_phone",                     False),
        ("home_phone",                      "cell_phone",                       False),
        ("phone_primary",                   "phone_secondary",                  False),
        ("telefono_casa",                   "telefono_oficina",                 False),
        ("celular",                         "telefono_fijo",                    False),
        ("phoneTrue",                          "phone2",                           False),
        ("whatsapp",                        "telefono",                         False),

        # ══════════════════════════════════════════════════════
        # PRECIO: TIPO DISTINTO
        # ══════════════════════════════════════════════════════
        ("precio_venta",                    "precio_costo",                     False),
        ("precio_venta",                    "precio_compra",                    False),
        ("sale_price",                      "cost_price",                       False),
        ("unit_price",                      "unit_cost",                        False),
        ("list_price",                      "purchase_price",                   False),
        ("precio_minimo",                   "precio_maximo",                    False),
        ("precio_con_iva",                  "precio_sin_iva",                   False),
        ("precio_mayoreo",                  "precio_menudeo",                   False),
        ("costo_promedio",                  "costo_estandar",                   False),

        # ══════════════════════════════════════════════════════
        # STOCK MÍN / MÁX
        # ══════════════════════════════════════════════════════
        ("stock_minimo",                    "stock_maximo",                     False),
        ("minimum_stock",                   "maximum_stock",                    False),
        ("qty_min",                         "qty_max",                          False),
        ("reorder_point",                   "max_stock",                        False),

        # ══════════════════════════════════════════════════════
        # FECHAS DISTINTAS
        # ══════════════════════════════════════════════════════
        ("fecha_creacion",                  "fecha_modificacion",               False),
        ("fecha_inicio",                    "fecha_fin",                        False),
        ("fecha_inicio",                    "fecha_vencimiento",                False),
        ("fecha_emision",                   "fecha_vencimiento",                False),
        ("fecha_emision",                   "fecha_pago",                       False),
        ("fecha_nacimiento",                "fecha_creacion",                   False),
        ("fecha_nacimiento",                "fecha_contratacion",               False),
        ("fecha_entrega",                   "fecha_envio",                      False),
        ("start_date",                      "end_date",                         False),
        ("creation_date",                   "modification_date",                False),
        ("due_date",                        "payment_date",                     False),
        ("birth_date",                      "hire_date",                        False),
        ("invoice_date",                    "due_date",                         False),
        ("delivery_date",                   "ship_date",                        False),

        # ══════════════════════════════════════════════════════
        # DOCUMENTOS DISTINTOS
        # ══════════════════════════════════════════════════════
        ("folio_factura",                   "folio_pedido",                     False),
        ("folio_factura",                   "folio_cotizacion",                 False),
        ("folio_pedido",                    "folio_cotizacion",                 False),
        ("order_id",                        "invoice_id",                       False),
        ("quote_number",                    "order_number",                     False),
        ("po_number",                       "invoice_number",                   False),

        # ══════════════════════════════════════════════════════
        # UUID ≠ FOLIO (confusión clásica en CFDI)
        # ══════════════════════════════════════════════════════
        ("cfdi_uuid",                       "folio_factura",                    False),
        ("uuid_sat",                        "serie_factura",                    False),
        ("invoice_uuid_sat",                "invoice_number",                   False),
        ("timbre_uuid",                     "folio_interno",                    False),
        ("folio_fiscal",                    "folio_interno",                    False),
        ("custbody_mx_cfdi_uuid",           "custrecord_cfdi_serie",            False),

        # ══════════════════════════════════════════════════════
        # CAMPOS SAT QUE SE CONFUNDEN
        # ══════════════════════════════════════════════════════
        ("metodo_pago",                     "forma_pago",                       False),
        ("uso_cfdi",                        "regimen_fiscal",                   False),
        ("uso_cfdi",                        "metodo_pago",                      False),
        ("regimen_fiscal",                  "tipo_persona",                     False),
        ("cfdi_use_code",                   "tax_regime_code",                  False),
        ("payment_method_sat",              "payment_form_sat",                 False),

        # ══════════════════════════════════════════════════════
        # IDs FISCALES DISTINTOS
        # ══════════════════════════════════════════════════════
        ("rfc",                             "curp",                             False),
        ("rfc",                             "nss",                              False),
        ("curp",                            "nss",                              False),
        ("tax_id",                          "social_security_number",           False),
        ("rfc",                             "clabe",                            False),
        ("rfc",                             "uuid",                             False),

        # ══════════════════════════════════════════════════════
        # FINANCIERO: CAMPOS RELACIONADOS PERO DISTINTOS
        # ══════════════════════════════════════════════════════
        ("subtotal",                        "total",                            False),
        ("subtotal",                        "descuento",                        False),
        ("total",                           "descuento",                        False),
        ("iva",                             "descuento",                        False),
        ("iva",                             "isr",                              False),
        ("tasa_iva",                        "tasa_isr",                         False),
        ("net_amount",                      "gross_amount",                     False),
        ("tax_amount",                      "discount_amount",                  False),
        ("clabe",                           "cuenta_bancaria",                  False),
        ("iban",                            "clabe",                            False),
        ("monto_pagado",                    "monto_pendiente",                  False),
        ("saldo_favor",                     "saldo_cargo",                      False),

        # ══════════════════════════════════════════════════════
        # PRODUCTO: DISTINTOS CAMPOS
        # ══════════════════════════════════════════════════════
        ("sku",                             "upc",                              False),
        ("sku",                             "ean",                              False),
        ("upc",                             "ean",                              False),
        ("product_code",                    "product_name",                     False),
        ("nombre_producto",                 "descripcion_producto",             False),
        ("codigo_producto",                 "descripcion_producto",             False),
        ("item_code",                       "item_description",                 False),
        ("sku",                             "nombre_producto",                  False),

        # ══════════════════════════════════════════════════════
        # CAMPOS COMPLETAMENTE DISTINTOS
        # ══════════════════════════════════════════════════════
        ("email",                           "telefono",                         False),
        ("email",                           "codigo_postal",                    False),
        ("email",                           "fecha_nacimiento",                 False),
        ("nombre",                          "rfc",                              False),
        ("nombre",                          "fecha_nacimiento",                 False),
        ("sku",                             "email",                            False),
        ("precio_venta",                    "fecha_vencimiento",                False),
        ("uuid",                            "celular",                          False),
        ("colonia",                         "iva",                              False),
        ("nombre_producto",                 "cuenta_bancaria",                  False),
        ("fecha_emision",                   "celular",                          False),
        ("folio_pedido",                    "email",                            False),
        ("rfc",                             "precio_venta",                     False),
        ("curp",                            "stock",                            False),
        ("clabe",                           "nombre",                           False),

        # ══════════════════════════════════════════════════════
        # SECUENCIALES (True vs 2)
        # ══════════════════════════════════════════════════════
        ("phoneTrue",                          "phone2",                           False),
        ("addressTrue",                        "address2",                         False),
        ("emailTrue",                          "email2",                           False),
        ("contact_person_True",                "contact_person_2",                 False),
        ("referenceTrue",                      "reference2",                       False),
        ("cuenta_principal",                "cuenta_secundaria",                False),

        # ══════════════════════════════════════════════════════
        # EDGE CASES: MUY SIMILARES PERO DISTINTOS
        # ══════════════════════════════════════════════════════
        ("customer_email_verified",         "customer_email",                   False),
        ("rfc_temporal",                    "rfc_definitivo",                   False),
        ("phone_primary",                   "phone_secondary",                  False),
        ("precio_con_iva",                  "precio_sin_iva",                   False),
        ("tax_code",                        "product_code",                     False),
        ("country_code",                    "area_code",                        False),
        ("department_code",                 "employee_code",                    False),
        ("activo",                          "eliminado",                        False),
        ("is_active",                       "is_deleted",                       False),
        ("fecha_inicio",                    "fecha_fin",                        False),
    ]


