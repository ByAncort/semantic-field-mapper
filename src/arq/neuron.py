import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AbsoluteValue(layers.Layer):
    """Capa personalizada para calcular el valor absoluto de manera serializable."""
    
    def call(self, x):
        return tf.abs(x)
    
    def get_config(self):
        return super().get_config()


class Neuron():
    @staticmethod
    def dense_classifier(input_dim=12):
        """
        Clasificador denso tradicional para vectores de features planos.

        Usado por el pipeline clásico que extrae 11-12 features por par de campos.

        Args:
            input_dim: Dimensión del vector de entrada (11 o 12 features)

        Returns:
            Modelo compilado listo para entrenamiento
        """
        modelo = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid', name='is_match')
        ])
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.003),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return modelo
    @staticmethod
    def simple_model(vocab_size=10000, embed_dim=64, max_len=20):
        """
        Modelo Siamés simple: compara dos campos y predice si hacen match.
        Input: [src_name, src_value, tgt_name, tgt_value]
        """
        # Encoder compartido
        field_input = keras.Input(shape=(max_len,))
        x = layers.Embedding(vocab_size, embed_dim)(field_input)
        x = layers.GlobalAveragePooling1D()(x)
        encoder = keras.Model(inputs=field_input, outputs=x)

        # 4 inputs: nombre y valor de campo origen y destino
        src_name  = keras.Input(shape=(max_len,), name='src_name')
        src_value = keras.Input(shape=(max_len,), name='src_value')
        tgt_name  = keras.Input(shape=(max_len,), name='tgt_name')
        tgt_value = keras.Input(shape=(max_len,), name='tgt_value')

        # Encodear y combinar cada campo
        src = layers.Concatenate()([encoder(src_name), encoder(src_value)])
        tgt = layers.Concatenate()([encoder(tgt_name), encoder(tgt_value)])

        # Medir similitud
        diff    = layers.Subtract()([src, tgt])
        merged  = layers.Concatenate()([src, tgt, diff])

        x = layers.Dense(32, activation='relu')(merged)
        output = layers.Dense(1, activation='sigmoid', name='is_match')(x)

        modelo = keras.Model(
            inputs=[src_name, src_value, tgt_name, tgt_value],
            outputs=output
        )
        modelo.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return modelo

    @staticmethod
    def deep_model(vocab_size=10000, embed_dim=64, max_len=20):
        """
        Modelo Siamés profundo: además detecta el TIPO de match.
        match_type → 0: no_match | 1: exacto | 2: composición | 3: transformación
        """
        # Encoder compartido más profundo
        field_input = keras.Input(shape=(max_len,))
        x = layers.Embedding(vocab_size, embed_dim)(field_input)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        encoder = keras.Model(inputs=field_input, outputs=x)

        # 4 inputs
        src_name = keras.Input(shape=(max_len,), name='src_name')
        src_value = keras.Input(shape=(max_len,), name='src_value')
        tgt_name = keras.Input(shape=(max_len,), name='tgt_name')
        tgt_value = keras.Input(shape=(max_len,), name='tgt_value')

        # Encodear y combinar
        src = layers.Concatenate()([encoder(src_name), encoder(src_value)])
        tgt = layers.Concatenate()([encoder(tgt_name), encoder(tgt_value)])

        # Capturar similitud y diferencia
        diff = layers.Subtract()([src, tgt])
        product = layers.Multiply()([src, tgt])
        merged = layers.Concatenate()([src, tgt, diff, product])

        x = layers.Dense(64, activation='relu')(merged)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Salida 1: ¿hace match?
        match_output = layers.Dense(1, activation='sigmoid', name='is_match')(x)
        # Salida 2: ¿qué tipo de match?
        type_output = layers.Dense(4, activation='softmax', name='match_type')(x)

        deep_model = keras.Model(
            inputs=[src_name, src_value, tgt_name, tgt_value],
            outputs=[match_output, type_output]
        )
        deep_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'is_match': 'binary_crossentropy',
                'match_type': 'sparse_categorical_crossentropy'
            },
            loss_weights={'is_match': 1.0, 'match_type': 0.5},
            metrics={
                'is_match': ['accuracy', 'AUC'],
                'match_type': ['accuracy']
            }
        )
        return deep_model

    @staticmethod
    def semantic_model(vocab_size=10000, embed_dim=64, max_len=20, semantic_features=9):
        """
        Modelo Semántico Avanzado: Integra análisis semántico con arquitectura Siamesa.

        Inputs:
            - src_name, src_value, tgt_name, tgt_value: campos a comparar (como antes)
            - src_semantic: features semánticas del campo origen (diferenciadores, identidad, grupo)
            - tgt_semantic: features semánticas del campo destino
            - context_features: features de contexto JSON (profundidad, tipo_objeto, etc.)

        Outputs:
            - similarity_score: probabilidad de match (0.0 - 1.0)
            - match_type: tipo de match (0: no_match, 1: exacto, 2: composición, 3: transformación, 4: derivado)
            - confidence: confianza de la predicción
        """

        # ============================================================
        # RAMA 1: PROCESAMIENTO DE TEXTO (Encoder Siamés)
        # ============================================================
        field_input = keras.Input(shape=(max_len,))
        x = layers.Embedding(vocab_size, embed_dim)(field_input)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Attention()([x, x])  # Auto-atención para capturar relaciones internas
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        text_encoder = keras.Model(inputs=field_input, outputs=x)

        # ============================================================
        # RAMA 2: CARACTERÍSTICAS SEMÁNTICAS
        # ============================================================
        semantic_input = keras.Input(shape=(semantic_features,), name='semantic_features')
        s = layers.Dense(32, activation='relu')(semantic_input)
        s = layers.BatchNormalization()(s)
        s = layers.Dropout(0.2)(s)
        s = layers.Dense(16, activation='relu')(s)
        semantic_processor = keras.Model(inputs=semantic_input, outputs=s)

        # ============================================================
        # RAMA 3: FEATURES DE CONTEXTO
        # ============================================================
        context_input = keras.Input(shape=(5,), name='context_features')
        # 5 features: profundidad, is_array, is_object, parent_type, position_en_objeto
        c = layers.Dense(16, activation='relu')(context_input)
        c = layers.BatchNormalization()(c)
        context_processor = keras.Model(inputs=context_input, outputs=c)

        # ============================================================
        # INPUTS PRINCIPALES
        # ============================================================
        src_name = keras.Input(shape=(max_len,), name='src_name')
        src_value = keras.Input(shape=(max_len,), name='src_value')
        tgt_name = keras.Input(shape=(max_len,), name='tgt_name')
        tgt_value = keras.Input(shape=(max_len,), name='tgt_value')
        src_semantic = keras.Input(shape=(semantic_features,), name='src_semantic')
        tgt_semantic = keras.Input(shape=(semantic_features,), name='tgt_semantic')
        context = keras.Input(shape=(5,), name='context')

        # ============================================================
        # PROCESAMIENTO: CODIFICAR CAMPOS DE TEXTO
        # ============================================================
        src_name_enc = text_encoder(src_name)
        src_value_enc = text_encoder(src_value)
        src_text = layers.Concatenate()([src_name_enc, src_value_enc])

        tgt_name_enc = text_encoder(tgt_name)
        tgt_value_enc = text_encoder(tgt_value)
        tgt_text = layers.Concatenate()([tgt_name_enc, tgt_value_enc])

        # ============================================================
        # PROCESAMIENTO: INTEGRAR FEATURES SEMÁNTICAS
        # ============================================================
        src_sem_proc = semantic_processor(src_semantic)
        tgt_sem_proc = semantic_processor(tgt_semantic)
        context_proc = context_processor(context)

        src_combined = layers.Concatenate()([src_text, src_sem_proc, context_proc])
        tgt_combined = layers.Concatenate()([tgt_text, tgt_sem_proc, context_proc])

        # ============================================================
        # COMPARACIÓN Y SIMILITUD
        # ============================================================
        # Similitud coseno implícita mediante resta y producto elemento a elemento
        diff = layers.Subtract()([src_combined, tgt_combined])
        product = layers.Multiply()([src_combined, tgt_combined])

        # Manhattan distance usando la capa personalizada serializable
        abs_diff = AbsoluteValue()(diff)

        # Concatenar todas las representaciones
        merged = layers.Concatenate()([
            src_combined,
            tgt_combined,
            diff,
            product,
            abs_diff
        ])

        # ============================================================
        # CAPAS DENSAS CON REGULACIÓN
        # ============================================================
        x = layers.Dense(128, activation='relu')(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # ============================================================
        # SALIDAS MÚLTIPLES
        # ============================================================
        # Salida 1: Score de similitud (0.0 - 1.0)
        similarity = layers.Dense(1, activation='sigmoid', name='similarity_score')(x)

        # Salida 2: Tipo de match (multi-clase)
        match_type = layers.Dense(5, activation='softmax', name='match_type')(x)

        # Salida 3: Confianza de la predicción
        confidence = layers.Dense(1, activation='sigmoid', name='confidence')(x)

        # ============================================================
        # COMPILACIÓN DEL MODELO
        # ============================================================
        semantic_model = keras.Model(
            inputs=[
                src_name, src_value, tgt_name, tgt_value,
                src_semantic, tgt_semantic,
                context
            ],
            outputs=[similarity, match_type, confidence]
        )

        semantic_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss={
                'similarity_score': 'binary_crossentropy',
                'match_type': 'sparse_categorical_crossentropy',
                'confidence': 'mse'
            },
            loss_weights={
                'similarity_score': 1.0,
                'match_type': 0.7,
                'confidence': 0.3
            },
            metrics={
                'similarity_score': ['accuracy', 'AUC', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                'match_type': ['accuracy'],
                'confidence': ['mae']
            }
        )

        return semantic_model