import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()
    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
    def draw_page_number(self, page_count):
        if self._pageNumber == 1: return
        self.saveState()
        self.setFont('Helvetica', 9)
        self.setFillColor(colors.HexColor('#555555'))
        self.drawString(54, 750, 'PROYECTO INTEGRAL IA: SISTEMA DE SOPORTE A LA DECISIÓN (DSS)')
        self.setStrokeColor(colors.HexColor('#CCCCCC'))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        page_text = f'Página {self._pageNumber} de {page_count}'
        self.drawRightString(558, 40, page_text)
        self.restoreState()

def crear_pdf():
    doc = SimpleDocTemplate('Informe_Proyecto_IA_Restaurantes.pdf', pagesize=letter, rightMargin=54, leftMargin=54, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CoverTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=24, leading=30, textColor=colors.HexColor('#1A365D'), alignment=1)
    subtitle_style = ParagraphStyle('CoverSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=13, leading=17, textColor=colors.HexColor('#4A5568'), alignment=1)
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=15, leading=19, textColor=colors.HexColor('#2B6CB0'), spaceBefore=16, spaceAfter=8, keepWithNext=True)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14.5, textColor=colors.HexColor('#2D3748'), spaceAfter=10, alignment=4)

    story = [
        Spacer(1, 150), 
        Paragraph('PROYECTO INTEGRAL DE IA:<br/>SISTEMA DE SOPORTE A LA DECISIÓN (DSS)', title_style), 
        Spacer(1, 15), 
        Paragraph('Recomendador Predictivo Gastronómico mediante Algoritmo SVD', subtitle_style), 
        Spacer(1, 200), 
        Paragraph('<b>Autor:</b> Diana<br/><b>Curso:</b> Inteligencia Artificial Aplicada<br/><b>Entorno:</b> Python 3.11 / Surprise Framework', ParagraphStyle('Meta', fontName='Helvetica', fontSize=11, leading=16, textColor=colors.HexColor('#718096'), alignment=1)), 
        PageBreak()
    ]

    sections = [
        ('1. Definición del Problema (10%)', 'En el sector de servicios moderno, los usuarios se enfrentan diariamente a la \"parálisis por análisis\". La sobreabundancia de opciones gastronómicas satura los mecanismos de elección tradicionales, provocando insatisfacción. Las plataformas convencionales limitan sus interfaces a filtros heurísticos rígidos (orden alfabético o cercanía), ignorando el comportamiento histórico del cliente. El objetivo es diseñar un Sistema de Soporte a la Decisión (DSS) impulsado por IA que mitigue la fricción mediante la predicción estocástica de la utilidad esperada de un restaurante para un usuario específico, optimizando la experiencia del consumidor final.'),
        ('2. Fundamentos y Uso de Aplicaciones Comunes de IA (10%)', 'Se analizaron dos ramas del ecosistema de IA: Sistemas basados en Contenido (utilizan vectores de características de los ítems, pero sufren de sobreespecialización) y Filtrado Colaborativo Basado en Modelos (utiliza el comportamiento cruzado histórico de la comunidad para inferir afinidades ocultas). Este último enfoque fue el seleccionado para el DSS. Al transformar una base de datos relacional inerte en una matriz dinámica de utilidad, se logran predecir interacciones futuras con alta precisión matemática sin requerir perfiles manuales descriptivos.'),
        ('3. Aplicaciones y Herramientas de Usuario IA (10%)', 'La implementación técnica se ejecutó sobre un ecosistema Open Source en Python utilizando Surprise (motor optimizado para el manejo de matrices dispersas y entrenamiento SVD) y Pandas (motor estructurado para procesos de ETL). La integración consolidó una arquitectura desacoplada donde Pandas actúa como el validador estricto de tipos de datos de entrada mediante coerciones de error, inyectando una matriz limpia de tres columnas fundamentales [user_id, restaurant_id, rating] dentro de la infraestructura del modelo predictivo.'),
        ('4. Escenarios Prácticos y Estudios de Caso (10%)', 'Un caso de estudio relevante es el algoritmo CineMatch de Netflix en su fase inicial, así como las arquitecturas híbridas de Yelp. Estas plataformas migraron de listados estructurados por categorías fijas a matrices densas de utilidad calculadas por lotes. La lección clave aplicada a nuestro proyecto es el establecimiento explícito de un objeto Reader que normaliza las fronteras de optimización del algoritmo, garantizando distribuciones acotadas y evitando sesgos provocados por calificaciones extremas en la comunidad.'),
        ('5. Algoritmos y Complejidad en IA (10%)', 'El núcleo predictivo implementa la Descomposición en Valores Singulares (SVD), la cual factoriza la matriz original de utilidad de tamaño M x N en matrices latentes de menor rango que capturan un número K de factores esenciales mediante la ecuación de optimización de sesgos interdependientes. Su complejidad temporal de entrenamiento mediante Descenso de Gradiente Estocástico (SGD) es lineal respecto al número de muestras, mientras que el tiempo de inferencia es de orden constante O(1), garantizando que el DSS responda al cliente de manera instantánea.'),
        ('6. Reconocimiento de Imágenes y Procesamiento de Lenguaje Natural (10%)', 'En el contexto del pipeline del DSS, el procesamiento de texto actúa como la capa heurística inicial. Se implementó una normalización completa sobre las columnas de texto (cuisine_type y restaurant_name) eliminando espacios vacíos perimetrales y forzando minúsculas. Esto dota de robustez al sistema ante cadenas mal estructuradas o caracteres corruptos detectados en el archivo original de transacciones, como se demostró al corregir y limpiar dinámicamente los registros incompletos del dataset de restaurantes.'),
        ('7. IA en la Toma de Decisiones (10%)', 'La IA transforma la toma de decisiones al sustituir la aproximación empírica intuitiva del cliente por una metodología predictiva basada en datos masivos. El DSS genera valor agregado al presentar al usuario un \"Reporte de Decisión Optimizado\". Al ordenar las alternativas en función de la afinidad latente estimada por el modelo SVD, se reduce el costo cognitivo de elección, disminuyendo radicalmente las tasas de abandono de las plataformas de búsqueda tradicionales.'),
        ('8. Desafíos y Limitaciones de la IA (10%)', 'El proyecto enfrenta dos desafíos inherentes clásicos: el problema de arranque en frío (Cold Start) ante usuarios nuevos sin historial de calificaciones y la dispersión de la matriz (Sparsity). Como estrategia de mitigación, se diseñó e integró un modelo híbrido adaptativo que activa un algoritmo heurístico por popularidad global (Función 1) ante la detección de un usuario nuevo, migrando suavemente al modelo predictivo puro SVD (Función 2) tras recolectar sus primeras interacciones.'),
        ('9. Ética en IA y su Impacto Social y Económico (10%)', 'El filtrado colaborativo introduce riesgos éticos como la creación de \"burbujas de filtro\" que aíslan el consumo. A nivel económico, un sesgo en el modelo podría invisibilizar de forma sistemática a nuevos emprendimientos gastronómicos pequeños que carecen de un volumen inicial alto de reseñas, concentrando los flujos de capital en operadores establecidos. El barajado estocástico aleatorio controlado es esencial para mantener la equidad competitiva en la plataforma.'),
        ('10. IA en Empleabilidad (10%)', 'La adopción de sistemas DSS basados en IA redefine los perfiles laborales del sector gastronómico y turístico. Se reduce la demanda de analistas de marketing tradicionales encargados del perfilado manual estático, desplazando el valor laboral hacia ingenieros de datos y especialistas en MLOps. Se recomienda capacitar al personal actual en la interpretación de métricas analíticas, transformando el potencial desplazamiento en una transición hacia puestos hiper-especializados de supervisión algorítmica.'),
        ('11. Dirección de Acceso', 'Network URL: http://192.168.1.130:8501 , puedes visitar la url del sitio.')
    ]

    for heading, text in sections:
        story.append(Paragraph(heading, h1_style))
        story.append(Paragraph(text, body_style))
        story.append(Spacer(1, 4))

    doc.build(story, canvasmaker=NumberedCanvas)
    print('\n✅ ¡Archivo \"Informe_Proyecto_IA_Restaurantes.pdf\" generado con éxito!')

if __name__ == '__main__':
    crear_pdf()