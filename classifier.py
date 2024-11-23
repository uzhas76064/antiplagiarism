import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Пример данных
data = [
    (
    'Перечисленные свойства и характеристики простейшего потока широко используются при расчетах станционного оборудования и сетей связи.',
    'Man'),
    (
    'The listed properties and characteristics of the simplest flow are widely used in calculations of station equipment and communication networks.',
    'Man'),
    (
    'Модель простейшего потока получила наибольшее по отношению ко всем другим моделям распространение в теории телетрафика.',
    'Man'),
    (
    'The simplest flow model has received the greatest distribution in relation to all other models in the theory of teletraffic.',
    'Man'),
    (
    'Данная модель позволяет достаточно хорошо описать реальный нестационарный поток вызовов, например, поступление вызовов на телефонную станцию в течение суток.',
    'Man'),
    (
    'This model allows us to describe quite well the real non-stationary call flow, for example, the arrival of calls to a telephone exchange during the day.',
    'Man'),
    (
    'На интенсивность нагрузки влияют следующие факторы: структурный состав абонентов (доля квартирного, народнохозяйственного и административного секторов), ритм местной жизни (начало и конец рабочего дня), время суток, день недели, число месяца, месяц года.',
    'Man'),
    (
    'The load intensity is influenced by the following factors: structural composition of subscribers (share of residential, national economic and administrative sectors), the rhythm of local life (beginning and end of the working day), time of day, day of the week, day of the month, month of the year.',
    'Man'),
    (
    'При исследовании суточного распределения нагрузки выделяют промежуток длиной в 1 час, когда нагрузка достигает своего максимального значения.',
    'Man'),
    (
    'When studying the daily load distribution, a period of 1 hour is identified when the load reaches its maximum value.',
    'Man'),
    (
    'Чем меньше коэффициент концентрации, тем равномерней загружено коммутационное оборудование, тем меньше его объема требуется для выполнения одной и той же работы.',
    'Man'),
    (
    'The lower the concentration coefficient, the more evenly the switching equipment is loaded, the less its volume is required to perform the same work.',
    'Man'),
    (
    'Полнодоступной коммутационной системой называется система, в которой каждому входу доступен любой свободный выход.',
    'Man'),
    ('A fully accessible switching system is a system in which every input has access to every free output.', 'Man'),
    (
    'Вероятность потери вызова, поступившего в некотором промежутке времени, равная отношению средних интенсивностей потоков потерянных и поступивших вызовов в этом промежутке.',
    'Man'),
    (
    'The probability of losing a call received in a certain period of time is equal to the ratio of the average intensities of the flows of lost and received calls in this period.',
    'Man'),
    (
    'Итак, с помощью первой формулы Эрланга можно вычислить характеристики качества обслуживания полнодоступной системы с явными потерями, когда на неё поступает простейший поток вызовов.',
    'Man'),
    (
    'So, using the first Erlang formula, you can calculate the quality of service characteristics of a fully accessible system with obvious losses when it receives the simplest call flow.',
    'Man'),
    ('На полнодоступную систему с явными потерями поступает примитивный поток вызовов.', 'Man'),
    ('A fully accessible system with obvious losses receives a primitive call flow.', 'Man'),
    (
    'Требуется определить вероятность ожидания поступившего вызова в очереди, математические ожидания длительности ожидания и длины очереди.',
    'Man'),
    (
    'It is required to determine the probability of waiting for an incoming call in the queue, the mathematical expectations of the waiting duration and the length of the queue.',
    'Man'),
    (
    'В полнодоступной системе с ожиданием величина поступающей нагрузки равна обслуженной, поскольку считается, что в этом случае потерянной нагрузки нет.',
    'Man'),
    (
    'In a fully available wait system, the amount of load arriving is equal to the load being served, since there is no load lost in this case.',
    'Man'),
    ('Модель Берке соответствует условиям работы одного управляющего устройства, например, маркера координатной АТС.',
     'Man'),
    (
    'The Berke model corresponds to the operating conditions of one control device, for example, a marker of a coordinated telephone exchange.',
    'Man'),
    ('Если вызовы обслуживаются несколькими управляющими устройствами, то используется модель Кроммелина.', 'Man'),
    ('If calls are handled by multiple control devices, the Crommelin model is used.', 'Man'),
    (
    'Неполнодоступное включение линий – это простой экономичный способ объединения мелких полнодоступных пучков в один крупный неполнодоступный, позволяющий повысить использование линий.',
    'Man'),
    (
    'Partial link inclusion is a simple, cost-effective way to combine small fully available bundles into one large partial access bundle to improve line utilization.',
    'Man'),
    (
    'В настоящее время принципы неполнодоступного включения нашли применение в системах мобильной радиосвязи при динамическом распределении каналов между сотами.',
    'Man'),
    (
    'Currently, the principles of partial inclusion have found application in mobile radio communication systems for the dynamic distribution of channels between cells.',
    'Man'),
    ('Поэтому метод актуален и сегодня, возможно, будет использован и в будущих системах телекоммуникаций.', 'Man'),
    ('Therefore, the method is relevant today and may be used in future telecommunications systems.', 'Man'),
    (
    'Ступенчатая схема - число выходов нагрузочных групп, обслуживаемых одной линией, различно и монотонно увеличивается с ростом номера выхода.',
    'Man'),
    (
    'Step scheme - the number of outputs of load groups served by one line increases differently and monotonically with increasing output number.',
    'Man'),
    ('В неполнодоступных схемах используют три типа включения линий: прямое, перехваченное и со сдвигом.', 'Man'),
    ('In incompletely accessible schemes, three types of line connection are used: direct, intercepted and shifted.',
     'Man'),
    ('При включении со сдвигом выходы одной группы соединяются с разноименными выходами других групп.', 'Man'),
    ('When switched on with a shift, the outputs of one group are connected to the opposite outputs of other groups.',
     'Man'),
    (
    'Данная система является полнодоступной, так как любому входу доступен любой свободный выход, но в такой схеме в соединении точек коммутации участвуют также промежуточные линии, которые могут быть заняты.',
    'Man'),
    (
    'This system is fully accessible, since any free output is available to any input, but in such a scheme, intermediate lines that may be occupied also participate in connecting switching points.',
    'Man'),
    ('Расчет показателей качества обслуживания для многозвенных систем – очень сложная задача.', 'Man'),
    ('Calculating quality of service indicators for multi-tier systems is a very difficult task.', 'Man'),
    (
    'Первая тенденция связана с тем, что большинство абонентов предъявляют все более жесткие требования к качеству обслуживания трафика.',
    'Man'),
    (
    'The first trend is due to the fact that the majority of subscribers are placing increasingly stringent demands on the quality of traffic service.',
    'Man'),
    (
    'В математической статистике всю изучаемую совокупность однородных элементов принято называть генеральной совокупностью.',
    'Man'),
    (
    'In mathematical statistics, the entire population of homogeneous elements being studied is usually called the general population.',
    'Man'),
    (
    'В качестве основной модели для разработки непараметрического алгоритма последовательной классификации рассмотрим задачу последовательного испытания двух выборок.',
    'Man'),
    (
    'As a basic model for developing a nonparametric sequential classification algorithm, consider the problem of sequential testing of two samples.',
    'Man'),
    (
    'При недостаточной априорной информации об исследуемом объекте целесообразно использовать непараметрические методы, для того чтобы получить более адекватную математическую модель конкретной физической ситуации для построения классификатора.',
    'Man'),
    (
    'If there is insufficient a priori information about the object under study, it is advisable to use nonparametric methods in order to obtain a more adequate mathematical model of a specific physical situation for constructing a classifier.',
    'Man'),
    ('Математические процедуры в данном случае намного проще, чем при параметрических методах классификации.', 'Man'),
    ('The mathematical procedures in this case are much simpler than with parametric classification methods.', 'Man'),
    ('Полученные количественные результаты контроля позволяют отображать их в наглядной и удобной для оператора форме.',
     'Man'),
    ('The obtained quantitative control results allow them to be displayed in a visual and operator-friendly form.',
     'Man'),
    (
    'Применение непараметрических методов классификации позволяет повысить достоверность и скорость принимаемых человеком решений.',
    'Man'),
    (
    'The use of nonparametric classification methods makes it possible to increase the reliability and speed of human decisions.',
    'Man'),
    ('С изменением психоэмоционального состояния у человека меняется поведение, голос, жесты, мимика.', 'Man'),
    (
    'With a change in a person’s psycho-emotional state, a person’s behavior, voice, gestures, and facial expressions change.',
    'Man'),
    ('Также в мозгу происходят процессы, которые можно зафиксировать специальной аппаратурой.', 'Man'),
    ('There are also processes occurring in the brain that can be recorded with special equipment.', 'Man'),
    (
    'Для автоматизации процесса распознавания психоэмоционального состояния необходимо извлечь признаки из распознаваемого образа.',
    'Man'),
    (
    'To automate the process of recognizing a psycho-emotional state, it is necessary to extract features from the recognized image.',
    'Man'),
    ('Извлечение признаков является одним из фундаментальных шагов в распознавании психоэмоционального состояния.',
     'Man'),
    ('Feature extraction is one of the fundamental steps in recognizing psycho-emotional states.', 'Man'),
    ('Предварительная обработка показаний необходима для фильтрации шума от измерений.', 'Man'),
    ('Pre-processing of readings is necessary to filter noise from measurements.', 'Man'),
    (
    'Умея вычислять совместное распределение по стандартным0 алгоритмам, нетрудно рассчитать (хотя бы приближенно) и основные вероятностные характеристики обслуживания.',
    'Man'),
    (
    'Being able to calculate the joint distribution by standard algorithms, it is not difficult to calculate (at least approximately) the main probabilistic characteristics of service.',
    'Man'),
    (
    'Подход, основанный на внешнем виде, включает в себя предварительную обработку, за которой следует компактное кодирование посредством уменьшения статистической избыточности.',
    'Man'),
    (
    'The appearance-based approach involves preprocessing followed by compact encoding by reducing statistical redundancy.',
    'Man'),
    (
    'Предварительная обработка в большинстве случаев требуется для выравнивания геометрии на изображении лица, например, с помощью двух глаз и кончика носа в фиксированных положениях посредством деформации аффинной текстуры.',
    'Man'),
    (
    'Preprocessing in most cases is required to align the geometry in a face image, for example, by using two eyes and a nose tip in fixed positions through affine texture deformation.',
    'Man'),
    (
    'Оптический поток или вейвлета Габоры используются для захвата движения лица и надежной регистрации, соответственно, для успешного распознавания.',
    'Man'),
    (
    'Optical flow or wavelet Gabors are used to capture facial motion and robust registration accordingly for successful recognition.',
    'Man'),
    (
    'Форманта  это пик в частотном спектре, который является результатом резонансных частот любой акустической системы.',
    'Man'),
    ('A formant is a peak in the frequency spectrum that results from the resonant frequencies of any speaker system.',
     'Man'),
    ('Для человеческого голоса форманты распознаются как резонансные частоты голосовых путей.', 'Man'),
    ('For the human voice, formants are recognized as resonant frequencies of the vocal tract.', 'Man'),
    (
    'Формантные области не имеют прямого отношения к основной частоте и могут оставаться более или менее постоянными по мере фундаментальных изменений.',
    'Man'),
    (
    'Formant regions are not directly related to fundamental frequency and can remain more or less constant as fundamental changes occur.',
    'Man'),
    (
    'Если базовый уровень низок в диапазоне форманты, качество звука будет высоким, но, если фундаментальное значение выше областей формантности, звук будет слабым.',
    'Man'),
    (
    'If the fundamental level is low in the formant range, the sound quality will be high, but if the fundamental value is higher than the formant ranges, the sound will be weak.',
    'Man'),
    ('Речевой образец может быть смоделирован как линейная комбинация его прошлых образцов.', 'Man'),
    ('A speech pattern can be modeled as a linear combination of its past patterns.', 'Man'),
    (
    'Уникальный набор коэффициентов предикторов определяется путем минимизации суммы квадратов разностей между фактическими речевыми выборками и линейно предсказанными.',
    'Man'),
    (
    'A unique set of predictor coefficients is determined by minimizing the sum of squared differences between the actual speech samples and the linearly predicted ones.',
    'Man'),
    (
    'Общая проблема распознавания образов состоит из трех основных этапов: извлечение признаков, уменьшение признаков и классификация.',
    'Man'),
    (
    'The general pattern recognition problem consists of three main steps: feature extraction, feature reduction, and classification.',
    'Man'),
    ('Распознавание психоэмоционального состояния подпадает под основную категорию проблем распознавания образов.',
     'Man'),
    ('Recognition of psycho-emotional states falls under the main category of pattern recognition problems.', 'Man'),
    ('Таким образом, основные этапы распознавания образов здесь в равной степени применимы.', 'Man'),
    ('Thus, the basic steps of pattern recognition are equally applicable here.', 'Man'),
    (
    'Однако основная проблема в распознавании психоэмоционального состояния заключается в выделении существенных признаков и выборе подходящего классификатора.',
    'Man'),
    (
    'However, the main problem in recognizing a psycho-emotional state is identifying significant features and choosing an appropriate classifier.',
    'Man'),
    (
    'В статье рассмотрены подходы к извлечению признаков и особенностей выражения лица для распознавания психоэмоционального состояния.',
    'Man'),
    (
    'The article discusses approaches to extracting features and features of facial expression for recognizing a psycho-emotional state.',
    'Man'),
    (
    'Анализ дифракционных изображений, полученных в результате рассеивания широкополосных радиолокационных сигналов на объектах со сложной трехмерной поверхностью, во многом зависит от качества обработки информации.',
    'Man'),
    (
    'Analysis of diffraction images obtained as a result of scattering of broadband radar signals on objects with a complex three-dimensional surface largely depends on the quality of information processing.',
    'Man'),
    ('На формирование радиолокационного дифракционного изображения сказываются ряд случайных факторов.', 'Man'),
    ('A number of random factors affect the formation of a radar diffraction image.', 'Man'),
    ('Случайные факторы можно в зависимости от причин возникновения разделить на группы.', 'Man'),
    ('Random factors can be divided into groups depending on the causes of their occurrence.', 'Man'),
    (
    'Первая группа факторов определяется самим объектом - характером отражения и рассеивания сигнала от его поверхности.',
    'Man'),
    (
    'The first group of factors is determined by the object itself - the nature of the reflection and scattering of the signal from its surface.',
    'Man'),
    (
    'Ввиду структурной сложности дифракционного изображения объекта, многовариантностью ракурсов, внешнего оборудования и средств маскировки эта группа имеет детерминированную и случайную составляющие.',
    'Man'),
    (
    'Due to the structural complexity of the diffraction image of an object, the multivariance of angles, external equipment and camouflage means, this group has deterministic and random components.',
    'Man'),
    (
    'В теории распознавания используется термин “существенная размерность”, обозначающий минимальное число измерений (выделяемых признаков), необходимое для достаточно точной идентификации распознаваемых объектов.',
    'Man'),
    (
    'In recognition theory, the term “essential dimension” is used, denoting the minimum number of dimensions (identified features) necessary for a sufficiently accurate identification of recognized objects.',
    'Man'),
    (
    'Непересечение эллипсоидов рассеивания различных классов является необходимым условием при выборе эффективного набора признаков классификатора образов.',
    'Man'),
    (
    'The non-intersection of scattering ellipsoids of different classes is a necessary condition when choosing an effective set of features for an image classifier.',
    'Man'),
    ('При некотором упрощении задачи может быть получена оценка вероятности распознавания.', 'Man'),
    ('With some simplification of the problem, an estimate of the recognition probability can be obtained.', 'Man'),
    (
    'Это факт свидетельствует о том, что не правомерно использовать аппроксимацию многомерного эллипсоида соответствующей размерности параллелепипедом (как это делается в большинстве расчетов), так как в этом случае получается завышенная оценка вероятности, а доверительные интервалы имеют заниженную оценку.',
    'Man'),
    (
    'This fact indicates that it is not legal to use the approximation of a multidimensional ellipsoid of the appropriate dimension by a parallelepiped (as is done in most calculations), since in this case an overestimate of the probability is obtained, and the confidence intervals have an underestimate.',
    'Man'),
    (
    'Как правило, назначая уровень значимости признаков удается существенно понизить размерность этого пространства, например, используя упорядоченный набор собственных значений разложения Карунена-Лоева для ковариационной матрицы класса.',
    'Man'),
    (
    'As a rule, by assigning a significance level to features, it is possible to significantly reduce the dimension of this space, for example, using an ordered set of eigenvalues of the Karhunen-Loev decomposition for the class covariance matrix.',
    'Man'),
    (
    'Рассмотренные выше критерии качества информационных признаков могут быть положены в основу построения классификаторов на самых общих положениях статистической теории оценивания и теории информации и не используют конкретных методов и схем построения классификаторов.',
    'Man'),
    (
    'The criteria for the quality of information features discussed above can be used as the basis for constructing classifiers on the most general principles of statistical estimation theory and information theory and do not use specific methods and schemes for constructing classifiers.',
    'Man'),
    (
    'Изложенные в статье критерии составляют основу научно-обоснованной методики селекции информативных признаков классификаторов.',
    'Man'),
    (
    'The criteria set out in the article form the basis of a scientifically based methodology for selecting informative features of classifiers.',
    'Man'),
    (
    'Критерии отбора информационных признаков сложных пространственных объектов и сигналов могут быть применены для построения оптимального, в том числе, байесовского классификатора.',
    'Man'),
    (
    'Criteria for selecting information features of complex spatial objects and signals can be applied to build an optimal, including Bayesian, classifier.',
    'Man'),
    (
    'Решена задача автоматизации формирования выборок для построения диагностических и распознающих моделей по прецедентам.',
    'Man'),
    (
    'The problem of automating the formation of samples for constructing diagnostic and recognition models based on precedents has been solved.',
    'Man'),
    (
    'Предложен метод извлечения обучающих выборок, который обеспечивает сохранение в сформированной подвыборке важнейших топологических свойств исходной выборки, не требуя при этом загрузки в память ЭВМ исходной выборки, а также многочисленных проходов исходной выборки, что позволяет сократить объём выборки и уменьшить требования к ресурсам ЭВМ.',
    'Man'),
    (
    'A method for extracting training samples is proposed, which ensures that the most important topological properties of the original sample are preserved in the generated subsample, without requiring loading of the original sample into the computer memory, as well as numerous passes of the original sample, which makes it possible to reduce the sample size and reduce the requirements for computer resources.',
    'Man'),
    (
    'Однако, если изначально заданный набор признаков не является избыточным либо объем выборки чрезвычайно велик для представления и обработки в памяти ЭВМ, применение этих методов оказывается на практике затруднительным, а результаты их работы приводят к потере существенной для дальнейшего анализа информации либо не позволяют сохранить исходную интерпретабельность данных.',
    'Man'),
    (
    'However, if the initially specified set of features is not redundant or the sample size is extremely large for representation and processing in computer memory, the use of these methods turns out to be difficult in practice, and the results of their work lead to the loss of information essential for further analysis or do not allow maintaining the original interpretability of the data.',
    'Man'),
    (
    'Другим, существенно реже используемым на практике, подходом при решении данной задачи является сокращение объёма выборки.',
    'Man'),
    ('Another approach to solving this problem, much less commonly used in practice, is to reduce the sample size.',
     'Man'),
    (
    'Для обнаружения экземпляров, находящихся на границах классов, в общем случае необходимо решить задачу кластер-анализа, что требует определения расстояний между всеми экземплярами выборки.',
    'Man'),
    (
    'To detect instances located on class boundaries, in the general case it is necessary to solve the problem of cluster analysis, which requires determining the distances between all instances of the sample.',
    'Man'),
    (
    'Это, в свою очередь, требует либо загрузки всей выборки в память ЭВМ (что не всегда возможно из-за ограниченного объёма оперативной памяти), либо многократных проходов по исходной выборке (что вызывает значительные затраты машинного времени), а также приводит к необходимости хранить и обрабатывать матрицу расстояний между экземплярами большой размерности.',
    'Man'),
    (
    'This, in turn, requires either loading the entire sample into computer memory (which is not always possible due to the limited amount of RAM), or multiple passes through the original sample (which causes significant computer time consumption), and also leads to the need to store and process a high-dimensional matrix of distances between instances.',
    'Man'),
    (
    'Для устранения отмеченных недостатков предлагается заменить обработку экземпляров на обработку их описаний в виде числовых скаляров, которые характеризуют положение экземпляров в пространстве признаков.',
    'Man'),
    (
    'To eliminate the noted shortcomings, it is proposed to replace the processing of instances with the processing of their descriptions in the form of numerical scalars that characterize the position of instances in the feature space.',
    'Man'),
    (
    'Результаты проведенных экспериментов подтвердили работоспособность и практическую применимость предложенного метода, а также программного обеспечения, реализующего его.',
    'Man'),
    (
    'The results of the experiments confirmed the performance and practical applicability of the proposed method, as well as the software that implements it.',
    'Man'),
    (
    'В работе решена актуальная задача автоматизации формирования выборок для построения диагностических и распознающих моделей по прецедентам.',
    'Man'),
    (
    'The work solves the urgent problem of automating the formation of samples for building diagnostic and recognition models based on precedents.',
    'Man'),
    (
    'Научная новизна результатов работы заключается в том, что впервые предложен метод извлечения обучающих выборок, который обеспечивает сохранение в сформированной подвыборке важнейших для последующего анализа топологических свойств исходной выборки, не требуя при этом загрузки в память ЭВМ исходной выборки, а также многочисленных проходов по исходной выборке, что позволяет существенно сократить объём выборки, существенно уменьшить требования к ресурсам ЭВМ.',
    'Man'),
    (
    'The scientific novelty of the results of the work lies in the fact that for the first time a method for extracting training samples has been proposed, which ensures that the topological properties of the original sample that are most important for subsequent analysis are preserved in the generated subsample, without requiring loading of the original sample into the computer memory, as well as numerous passes through the original sample. which makes it possible to significantly reduce the sample size and significantly reduce the requirements for computer resources.',
    'Man'),
    (
    'Практическая значимость результатов работы состоит в том, что разработано программное обеспечение, реализующее предложенный метод формирования и редукции выборок, а также проведены эксперименты по их исследованию при решении практических задач, результаты которых позволяют рекомендовать разработанный метод для использования на практике при решении задач интеллектуального анализа данных.',
    'Man'),
    (
    'The practical significance of the results of the work is that software has been developed that implements the proposed method of forming and reducing samples, and experiments have been conducted to study them in solving practical problems, the results of which allow us to recommend the developed method for use in practice when solving problems of data mining.',
    'Man'),
    (
    'Дальнейшие исследования могут быть сосредоточены на разработке новых способов формирования описаний экземпляров в виде обобщённых показателей, разработке реализаций предложенного метода для параллельных вычислительных систем и распределенной обработки данных.',
    'Man'),
    (
    'Further research can be focused on developing new methods for generating descriptions of instances in the form of generalized indicators, developing implementations of the proposed method for parallel computing systems and distributed data processing.',
    'Man'),
    (
    'Недостатком известных технических решений является избыточность числа параметров модели, что затрудняет процесс их подбора и снижает обобщающие способности моделей.',
    'Man'),
    (
    'The disadvantage of the known technical solutions is the redundancy of the number of model parameters, which complicates the process of their selection and reduces the generalizing abilities of the models.',
    'Man'),
    (
    'Данная проблема возникает в связи с зависимостью числа параметров нейронной сети от числа входных переменных, поскольку в отличие от простых линейных моделей в нейронной сети связь между входными и выходными переменными осуществляется через, так называемые, скрытые слои нейронов, в которых осуществляется применение некоторой заданной передаточной функции к взвешенной сумме значений предыдущего слоя, в данном случае - входных признаков.',
    'Man'),
    (
    'This problem arises due to the dependence of the number of parameters of a neural network on the number of input variables, since, unlike simple linear models in a neural network, the connection between input and output variables is carried out through the so-called hidden layers of neurons, in which a certain specified transfer function is applied to the weighted sum of the values of the previous layer, in this case the input features.',
    'Man'),
    (
    'Нейронная сеть автоассоциативной памяти требует огромной обучающей выборки ввиду большого количества параметров, при этом в любом случае может проявляться отсутствие формальных ограничений на ее ответы и потенциальная опасность переучивания системы на одном типе дисторсии сигнала, после чего система будет пытаться устранить именно ее независимо от ее наличия или присутствия других типов шумов и смещений базовой линии.',
    'Man'),
    (
    'A neural network of auto-associative memory requires a huge training sample due to the large number of parameters, and in any case there may be a lack of formal restrictions on its responses and the potential danger of overtraining the system on one type of signal distortion, after which the system will try to eliminate it, regardless of its presence or presence of other types of noise and baseline shifts.',
    'Man'),
    (
    'Реализация с генетическим алгоритмом предполагает обучение нейронных сетей после каждой итерации добавления шума, говоря терминами генетических алгоритмов: после генерации поколения индивидуумов, т.е. набора выборок с несколько отличающимися параметрами распределения, на каждой из которых потребуется обучения нейронной сети.',
    'Man'),
    (
    'An implementation with a genetic algorithm involves training neural networks after each iteration of adding noise, in terms of genetic algorithms: after generating a generation of individuals, i.e. a set of samples with slightly different distribution parameters, each of which will require training of a neural network.',
    'Man'),
    ('Mother!', 'AI')
]
texts, labels = zip(*data)

# Преобразование текста в векторное представление
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Обучение наивного байесовского классификатора
model = MultinomialNB()
model.fit(X, labels)


# Функция оценки
def evaluate_model(test_data, gamma1=1, gamma2=1):
    test_texts, true_labels = zip(*test_data)
    X_test = vectorizer.transform(test_texts)
    predictions = model.predict(X_test)

    # Матрица ошибок
    matrix = confusion_matrix(true_labels, predictions)

    # Отчет классификации
    report = classification_report(true_labels, predictions, output_dict=True)

    # Агрегированные метрики
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    # Подробная статистика
    stats = {
        "Матрица ошибок": matrix,
        "Macro F1-Score": macro_f1,
        "Weighted F1-Score": weighted_f1,
        "Precision (macro avg)": precision,
        "Recall (macro avg)": recall,
        "Gamma1": gamma1,
        "Gamma2": gamma2,
    }
    return stats


# Графический интерфейс
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Классификатор текста")
        self.geometry("800x600")

        # Параметры
        self.gamma1_var = tk.DoubleVar(value=1)
        self.gamma2_var = tk.DoubleVar(value=1)

        # Поля ввода
        ttk.Label(self, text="Введите γ1:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(self, textvariable=self.gamma1_var).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self, text="Введите γ2:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(self, textvariable=self.gamma2_var).grid(row=1, column=1, padx=5, pady=5)

        # Кнопка анализа
        self.run_button = ttk.Button(self, text="Анализ", command=self.run_analysis)
        self.run_button.grid(row=2, column=0, columnspan=2, pady=10)

        # График
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.chart = FigureCanvasTkAgg(self.figure, self)
        self.chart.get_tk_widget().grid(row=3, column=0, columnspan=2)

        # Текстовая область для вывода статистики
        self.text_area = tk.Text(self, height=15, width=80)
        self.text_area.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

    def run_analysis(self):
        gamma1 = self.gamma1_var.get()
        gamma2 = self.gamma2_var.get()

        # Оценка метрик
        stats = evaluate_model(data, gamma1, gamma2)
        macro_f1 = stats["Macro F1-Score"]
        weighted_f1 = stats["Weighted F1-Score"]
        precision = stats["Precision (macro avg)"]
        recall = stats["Recall (macro avg)"]

        # Обновление графика
        self.ax.clear()
        self.ax.bar(["Macro F1", "Weighted F1", "Precision", "Recall"], [macro_f1, weighted_f1, precision, recall])
        self.ax.set_title("Метрики классификатора")
        self.ax.set_ylim(0, 1)
        self.chart.draw()

        # Вывод статистики в текстовую область
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, f"Матрица ошибок:\n{stats['Матрица ошибок']}\n\n")
        self.text_area.insert(tk.END, f"Macro F1-Score: {macro_f1:.2f}\n")
        self.text_area.insert(tk.END, f"Weighted F1-Score: {weighted_f1:.2f}\n")
        self.text_area.insert(tk.END, f"Precision (macro avg): {precision:.2f}\n")
        self.text_area.insert(tk.END, f"Recall (macro avg): {recall:.2f}\n")
        self.text_area.insert(tk.END, f"Gamma1: {stats['Gamma1']}\n")
        self.text_area.insert(tk.END, f"Gamma2: {stats['Gamma2']}\n")


# Запуск приложения
if __name__ == "__main__":
    app = App()
    app.mainloop()

