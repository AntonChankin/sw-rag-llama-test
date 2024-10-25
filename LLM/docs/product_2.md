# Система NAVIGATOR

## Производитель
Softwell

## Описание
NAVIGATOR —
Флагманский продукт компании Softwell, представляет собой единую платформу — интегрированное решение для всех рынков, обеспечивающее полный жизненный STP цикл казначейства на всех рынках — FX, Money Markets, Capital Markets, Financing, Interest Rate Derivatives и другие.
Решение охватывает все сферы жизнедеятельности казначейства: регистрацию сделок, управление позициями, расчет доходов, контроль лимитов, платежная позиция.
Для расчетов цен Навигатор использует онлайн интерфейсы к основным электронным торговым площадкам, включая RuTerminal, Московскую Биржу.

# Модули
## Рынки
### Валютный рынок
- Spot — покупка / продажа валют стандартными датами расчетов и на условиях Split value. Операции в рамках чистых линий, margin trading, операции клиентов по расчетным счетам в банке.
- Forward — покупка / продажа валют сроками Outright, поставочные и NDF. Возможность расчетов достаточности обеспеченности открытых позиций и стоимости замещения.
- Swaps
- Banknotes — операции с наличной валютой.

### Денежный рынок
Депозиты с фиксированной и плавающей ставкой:
- С графиком выплат / довложений основной суммы в течение сделки,
- С графиком изменения ставки в течение сделки,
- С графиком выплат и наращения процентов — например: ежеквартально, на даты IMM или в конце месяца, наращение процентов ежемесячно, еженедельно…
### Рынки ценных бумаг
- Покупка / продажа — биржевые и внебиржевые операции с акциями, облигациями и векселями,
- Репо — акции, облигации и корзины, с плавающей и фиксированной ставками.
- Займы (Securities lending) — акции, облигации.
### Производные
- Futures — на валюты, акции, индексы, товары, процентные ставки,
- Options — на валюты, акции, облигации, товары,
- IRS — процентные свопы OIS, XCCY, Basis
- Cap, Floor, Collars
- FRA — соглашения о будущей валютной ставке

## Казначейство
Модуль решает основные задачи казначейства — контроль ликвидности и управление процентными рисками банка.
### Контроль ликвидности
#### Краткосрочная ликвидность
В системе контроль краткосрочной ликвидности осуществляется с помощью платежного календаря, формируемого в режиме реального времени на основании операций торговых подразделений и клиентских платежей.
Любая сделка, зарегистрированная в системе формирует денежные потоки:
- В соответствии с платежными инструкциями банка, указанными в сделке или
- На основании стандартных платежных инструкций банка или торговых систем.
Календарь позволяет оценить ликвидность в разрезах расчетных счетов Банка или торговых книг Казначейства.
Календарь может быть приведен к той или иной валюте, что позволяет оценить общий избыток или дефицит ликвидности Банка на ближайшую перспективу.
Сотрудник казначейства может моделировать поведение ликвидности, регистрировать ожидаемые платежи, информации о которых еще нет в системах банка.
NAVIGATOR может принимать информацию о клиентских платежах из различных систем — из ИБС или платежных систем (SWIFT или МЦИ) с возможностями квитовки данных.
#### Среднесрочная ликвидность
Для анализа среднесрочной ликвидности в системе используются данные о срочных активах и пассивах банка — кредитах, репо, форвардных покупках и ожидаемых погашениях инструментов с фиксированной доходностью.
Отчеты казначейства позволяют оценить ликвидность в разрезах торговых книг и соответствующих им продуктов, валют или инструментов.
Данные о платежах группируются в стандартные корзины с возможностью расчета гэпов — для корзин и кумулятивного гэпа.
При наличии Лимит Сервера возможен контроль ограничений, устанавливаемых на разрывы ликвидности на тех или иных корзинах.
В расчеты могут быть включены экспертные данные об неснижаемых остатках по продуктам, сроки владения которыми не могут быть получены из данных по операциям банка, имеющим срочную природу.

### Контроль процентных рисков
#### Краткосрочная ликвидность
В системе реализован механизм внутреннего рынка ресурсов, для которого казначейство устанавливает трансфертные ставки.

Торговые книги подразделений объединяются в центры финансового учета (ЦФУ) с возможностью настройки их учетных политик, с указанием разрешенных продуктов и их ставок фондирования.

- Автоматическое фондирование сделок бизнес подразделений с помощью:
    1. Срочных сделок с торговыми книгами Казначейства по привлечению / размещению денежных средств
    2. Срочных сделок FX Swap — для фондирования открытых валютных позиций бизнес подразделений
- Полуавтоматическое создание сделок фондирования — сделки, заключенные на основании учетных позиций ЦФУ или переговоров между бизнес подразделениями и Казначейством — фондирование позиций, которые невозможно финансировать в автоматическом режиме.
На основании сделок фондирования формируется отчетность казначейства — процентный риск, процентные доходы, чувствительность доходов к изменению процентных ставок (Basic Point Value) для продуктов, содержащих процентный риск.

## Веб портал — инвестиционный кабинет
### Вeб портал является уникальным решением
На российском рынке ни одно из известных решений не предлагает одновременно все, что сделает для вас портал.
1. Клиент получает в свое распоряжение собственный NAVIGATOR — доступ к отчетам трейдера: торговые позиции, отчеты о доходах, рисках...
2. Доступ к отчетам банка, созданным с использованием конструктора отчетов и опубликованным в Веб портале
3. Регистрация операций в NAVIGATOR или иных системах банка — регистрация сделок, платежей.
    - Доступ к торговой платформе iWELL:
    - Торговля с банком (OTC), любые продукты — покупка валюты, свопы, депозиты, работа с векселями — все, чем располагает банк,
    - Торговля на бирже — ценные бумаги, покупка валюты, свопы, репо, ...
    - Торговля в окне FX ликвидности, собранной от лидеров межбанковских рынков.
### Отчеты
При установке портала банк получает в свое распоряжение типовой набор отчетов, повторяющий основной набор он-лайн интерфейсов трейдера в desktop приложении NAVIGATOR — валютная позиция, денежная позиция, позиция по ценным бумагам — для всех рынков, поддерживаемых NAVIGATOR.

В дополнение к продуктовым отчетам банк получает набор отчетов о сделках, платежах и балансах счетов — доступ к информации обо всех базовых объектах. Доступ к информации, как и в desktop приложении, определяется настройкой прав пользователя к торговым книгам и расчетным счетам банка.

Используя конструктор отчетов, банк может создавать собственные отчеты на основании данных любой их своих систем и публиковать их на портале — клиент или сотрудник банка может получать информацию об остатках на счетах в ИБС, остатках на пластиковых картах, платежах...

Причем, получать он их может в реальном времени и из любой точки мира, где есть интернет.

### Операции
Портал может использоваться для регистрации операций, как клиентами, так и сотрудниками удаленных офисов.

#### Клиент
Клиент может регистрировать стандартный набор документов — платежные поручения, переводы ценных бумаг, иные документы. Используются сертифицированные средства СКЗИ наших партнеров.
#### Удаленный офис, филиал
Филиалы получают в свое распоряжение полноценную систему управленческого учета операций, информация по ним доступна головному офису в режиме реального времени. Система может использоваться ими для регистрации сделок со своими клиентами на собственных торговых книгах или для регистрации заявок на перевод денежных средств для целей контроля ликвидности головным офисом. Используя iWELL, филиалы могут мгновенно закрывать свои позиции на головной офис.
### Торговля
Торговля подробно описана в разделе iWELL

### Кому это нужно:
#### Клиенту
Клиент получает в свое распоряжение собственный NAVIGATOR со всеми его он-лайн отчетами — возможностью контролировать свои позиции и доходы по ним, видеть остатки на своих счетах в разрезе мест хранения

Если у банка есть лицензия iWELL клиенты по Лоро, Margin trading, брокерским или генеральным договорам могут:

- Заключать сделки с банком
- Размещать и снимать ордера — биржевые и внебиржевые
- Давать распоряжения на вывод средств
Клиент — тоже трейдер, его интересуют умные отчеты.

Отчет по требованиям ФСФР, отправленный по почте Бэк офисом на следующий день к полудню, конечно, нужен, но — это вчерашний овернайт :)

#### Филиалу
Филиал получит в свое распоряжение настоящую систему управленческого учета, ему тоже нужно понимать, по каким курсам он может сейчас закрыть позицию, сколько он заработал за прошлую неделю, какой портфель кредитов у него сейчас и какова его средняя ставка или дюрация.

#### Трейдеру
Теперь трейдер всегда может знать, что происходит с позицией, ордерами и рынками — например, ответить на вопрос Руководителя, находясь вне офиса во время ланча — iPhone, iPad... или узнать — закрыт ли его ордер.

#### Руководителю
В любом месте, где есть интернет, вы можете видеть позиции, доходы — контролировать работу подразделения.
Теперь не нужно искать в почте отчет за прошлый квартал — достаточно запросить его в портале.

Отчеты руководителя могут быть созданы ИТ подразделением банка с использованием конструктора отчетов и быть доступными для него в портале по запросу или по расписанию.

## Отчеты
Сложность продуктов, с которыми работает инвестиционный департамент банка, увеличивается с каждым годом.

Несмотря на это информация о позициях, доходности должна быть представлена в простых и своевременных отчетах — легко и быстро получаемых в системе.

Навигатор обеспечивает следующие типы онлайн отчетов:

### Торговые позиции
- Объем и цена позиции
- Стоимость вложения, рыночная стоимость
### Платежные
- Платежный календарь по Ностро / Лоро счетам — агрегированный и в разрезах валют, счетов, типов операций
- Контроль исполнения платежей
### Доходы
- В разрезах продуктов, торговых книг,
- Различными методами
### Система позволяет в онлайн режиме получать все отчеты во многих разрезах:
- Филиалы / дочерние банки / компании,
- Профит центры,
- Портфели,
- Продукты,
- Торговые книги,
- Расчетные счета — платежный календарь — по денежным счетам и счетам депозитариев,
- Типы финансовых инструментов,
- Типы договоров — брокерские, ДУ, маржинальные, другие,
- Даты регистрации / различные даты валютирования,
- Контрагенты,
- И другие.
### Конструктор отчетов
- Мощная и исключительно гибкая система настройки отчетов на базе создаваемых Word и Excel шаблонов.
- Отчеты могу вызываться из NAVIGATOR, быть опубликованы в Веб портале или использоваться другими системами банка.

- Доступ к любым базам данных
- Возможность получения консолидированного отчета с данными из разных баз данных с возможностями Master/Detail связки наборов данных
- Система разграничения прав пользователей или групп пользователей к отчетам
- Гибко настраиваемое дерево отчетов
- Доступны мощные средства Excel, в том числе макросы VB
- Доступны встроенные средства группировок, сортировок, вывода итогов и т.п.
- Возможность работы с полученными данными непосредственно в стандартных таблицах NAVIGATOR с функциями прямого сохранения их в Excel, XML, а также сортировки и группировки данных непосредственно в таблицах
- Удобные средства импорта/экспорта отчетов
- Система может встраиваться в ПО сторонних разработчиков.
- Возможность сохранения подготовленных документов в базе данных NAVIGATOR
- Возможность сохранения подготовленных документов в файлах, имена которых определяются шаблонами
- Возможность отправки подготовленных документов по e-mail через SMTP или MAPI. Вам теперь не нужно готовить отчеты брокера по N клиeнтам и вручную раскладывать их в почту для отправки клиентам.
- В состав системы может входить Планировщик. В нем Вы можете создать любое количество заданий, включить в них любое количество отчетов и подготавливать отчеты по гибко настраиваемому расписанию. Подготовленные отчеты могут просто сохраняться в файлы, распечатываться или отправляться по e-mail подписчикам. К примеру, Вам теперь не обязательно каждое 5-е число каждого месяца готовить отчеты за прошедший месяц и отправлять их по e-mail руководителю. Достаточно один раз настроить соответствующее задание.

Интеграция
Любой проект NAVIGATOR невозможен без интеграции, обеспечивающей синхронизацию данных различных систем и гарантированную доставку данных между ними.

За многие годы работы мы создали адаптеры к различным торговым платформам и ИБС.Интеграционные работы пройдут быстро и без длительных согласований форматов передаваемых данных и логики взаимодействия систем.

На многих проектах внедрения мы смогли создать профессиональную команду, обладающую уникальным набором качеств:

Мы обладаем глубокими знаниями банковских технологий и являемся экспертами в области автоматизации банков,
Создана надежная корпоративная методология внедрения NAVIGATOR в уже существующей архитектуре банка.
Интеграционная платформа
Для целей интеграции используется наша платформа BUS, в основу которой положены следующие принципы:

Синхронный или асинхронный обмен данными между системами,
Гарантированная доставка информации с использованием механизма очередей — MSMQ,
Объединение, в том числе, уже внедрённых системы,
Системы ничего не знают о внутренней архитектуре и структуре баз данных других систем, как, впрочем, и о факте их наличия.
Торговые платформы
Прием котировок и сделок всех типов, публикация ордеров и потоков цен iWELL, прием платежей (Репо margin calls, вариационная маржа), событий исполнения сделок.

СофтВел может предложить готовые адаптеры к торговым платформам, обеспечивающие STP и он-лайн потоки котировок:

Ruterminal
MOEX
Autobahn
Barx
Refinitv
Bloomberg
Icap
JpMorgan
UBS
FXCM
При необходимости СофтВел может разработать адаптер к любой промышленной платформе в течение 3...5 недель.

Базы данных
Онлайн загрузка данных
Финансовые инструменты, купонные выплаты, погашения, оферты, дивиденды и рейтинги:

CBonds
Bloomberg
RuData
Бэк офисы
В случае внедрения NAVIGATOR без Бэк офиса мы можем предложить готовые адаптеры, обеспечивающие синхронизацию справочников и прием / передачу сделок и платежей для Бэк офисов компаний:

Diasoft
ЦФТ
Кворум
Новая Афина
Colvir
БИС
Адаптеры обеспечивают:
Двустороннюю синхронизация системных справочников бизнес партнеров, финансовых инструментов,
Двустороннее взаимодействие по передаче сделок всех типов, платежей и балансов.
При необходимости СофтВел может разработать адаптер к любому Бэк офису (включая in-house), имеющему требуемый API.

Платежные системы
Обработка входящих и исходящих платежных документов

Работа с различными типами документов:

МЦИ, SWIFT и КЦМР Казахстана,
Постановка входящих и исходящих платежей на Ностро позицию,
Прием и проверка подтверждающих документов (MT300/320).

Аппаратное обеспечение
Сервер базы данных
В качестве СУБД Resource NAVIGATOR XL использует:
Postgres Pro Enterprise версии 12
Oracle Database (Standard или Enderprise Edition) Server версий 12 или выше
Работа возможна на любой платформе, для которой сертифицированы СУБД.

Рекомендуемая конфигурация для 50 пользователей

Основной сервер
Сервер должен быть класса не ниже:
4xXeon 2,4Ghz
Оперативная память 8G
15-17G места на жестком диске из расчета на 1 млн сделок
Жесткие диски Raid 5+1 общий объем дискового пространства 300G, либо Storage.
2 блока питания горячей замены
Сеть 1G с pci резервом 100M
Backup сервер
Сервер должен быть класса не ниже:
Процессор 1 Xeon
Оперативная память 6-8G
Жесткие диски — на усмотрение банка
2 блока питания горячей замены
Сетевая плата
Примерные конфигурация основного сервера
Конфигурации серверов наших клиентов с высокой нагрузкой на систему
2x Intel(R) Xeon(R) CPU E5-2690 v2 @ 3.00GHz RAM 256 GB HDD 2 TB, несколько рейдов, данные, индексы, реду отдельно, реду на зеркале на ssd RHEL 2.6.18-371.el5 #1 SMP Thu Sep 5 21:21:44 EDT 2013 x86_64 x86_64 x86_64 GNU/Linux
DL380p Gen8 о двух 8-ядерных Xeon E5-2640 с 128 ГБ ОЗУ и СХД. Все под управлением RHEL6. Версия СУБД 11.2.0.3 Standard 64-bit.
Сервер торговой платформы iWELL
Требования для
50 активных клиентов
50 сделок в секунду
Сервер
Расположение — в сети банка.
Назначение — сервер, устанавливаемый в демилитаризованной зоне банка, обеспечивающий обмен информацией между клиентом и банком. Обмен информацией между сервером и клиентом осуществляется по протоколу TCP/IP.

Кроме этого приложения сервера могут работать в режиме шифрования трафика между сервером и клиентом. Значимые объекты подписываются ЭЦП (сделки, запросы на котировки, переговоры) с использованием средств криптозащиты.

Минимальные требования:
Жесткий диск 500 ГБ,
Скорость записи на жесткий диск не менее 100 Мбит/сек,
ОЗУ 8 ГБ,
Процессор 3 GHz,
ОС Windows Server 2012+,
Поддержка MSMQ,
Net Framework 4.7,
2 открытых порта (чтение / запись),
Доступ к БД,
Сервисы ценовой интеграции
Расположение — в сети банка.
Минимальные требования:
Жесткий диск 100 ГБ,
Скорость записи на жесткий диск не менее 100 Мбит/сек,
ОЗУ 4 ГБ,
Процессор 2 GHz,
ОС Windows Server 2012+,
Доступ к серверу iWell,
Net Framework 4.7,
Доступ к БД (При использовании спeцифичных сценариев загрузки сделок),
Порт для входящих подключений.
Клиентская часть
Требования к рабочему месту пользователей платформы iWELL.
Доступ к системе NavigatorWeb,
Наличие поддерживаемых браузеров:
Internet Explorer 11 и выше,
Google Chrome.
Рабочее место дилера
Требования к рабочему месту дилеров платформы iWELL:
Процессор Intel Core i5 или аналогичный,
ОЗУ от 4096 МБ
Наличие поддерживаемых браузеров:
Internet Explorer 11 и выше,
Google Chrome.

Стоимость лицензий и ТО
Стоимость лицензий
Стоимость лицензий можно узнать, связавшись с нами.

Для Связи используйте данные указанные внизу страницы.

Стоимость ТО
Обычно стоимость ТО определяется на основании стоимости лицензии и внедрения.

В ТО включаются следующие работы:
Исправление ошибок ПС, выяснение и устранение нештатных ситуаций (в любом количестве), а так же устранение последствий ошибок ПС в соответствии с Регламентом оказания услуг текущего сопровождения.
Адаптацию и модификацию ПС в соответствии с изменяющимися требованиями официальных организаций в соответствии с Регламентом оказания услуг текущего сопровождения
Передача Заказчику выпущенных за период сопровождения модифицированных версий ПС. Объём функциональности новой версии не должен быть меньше, чем в текущей версии ПС и полностью включать функционал текущей версии.
Необходимое количество консультаций (в том числе устных) по вопросам текущей эксплуатации ПС Заказчиком, использования банковских технологий в рамках ПС (с 09-00 по 19-00 московского времени в рабочие дни).
Предоставление Заказчику изменений Пользовательской документации по ПС в электронном виде.
Бесплатная разработка функционала по требованию Заказчика, если заказываемый функционал не противоречит архитектуре и расширяет функциональность системы.
В большинстве случаев годовая стоимость ТО не превышает 33% от суммы:

Cтоимости пятилетней лицензии,
Стоимости внедрения ПО.