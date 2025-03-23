import os
import argparse
import json

import openai

API_KEY = "" # OpenAI API Key
openai.api_key = API_KEY

OUTPUT_DIR = 'chatgpt/outputs'

examples_train = {
    'qalb14': [
        ('بسم الله والصلاة والسلام علي رسول الله وبعد : حقيقة أنا من محبي البرنامج الحية لأستمع الي بعض الحلقات الماضية اللواتى لم أشاهدهن مثل الاتجاه المعاكس . وحلقة أكثر من رأي ولكن عندما أشاهدها علي نت يكون فيها انقطاع الصوة أعتذر للكم أن تعالجه وشكرا موصول الكم', 'بسم الله - والصلاة والسلام على رسول الله - وبعد : حقيقة ، أنا من محبي البرامج الحية ، أستمع إلى بعض الحلقات الماضية التي لم أشاهدها مثل الإتجاه المعاكس ، وحلقة أكثر من رأي ، ولكن عندما أشاهدها على النت يكون فيها انقطاع الصوت ، أقترح عليكم أن تعالجوه ، وشكري موصول إليكم .'),
        ('الله مبارك للقاعدة كم بركتا على ابراهيم والي ابراهيم . واتمنا من قناة الجزيرة ان لاتنقلب لاجلي ارضاء السعودية . وكم شهدنا مراسل الجزيرة هذا العام بث على المباشر ايام العيد من مكة المكرمة . وهذا دليل انقلاب الجزيرة في هاذه الايام نرها تلعب على اعصاب المسلمين وخاصة انصار القاعدة واحبائها وهم والحمد لله بي الملاين . يا جزيرة الحق ينتصر فلاتنقلبي . وعيد سعيد لكل مسلم يحب الله', 'الله مبارك للقاعدة . كم بركة على إبراهيم وإلى إبراهيم ! وأتمنى من قناة الجزيرة أن لا تنقلب لأجل إرضاء السعودية . وكم شاهدنا مراسل الجزيرة هذا العام يبث على المباشر أيام العيد من مكة المكرمة ! وهذا دليل انقلاب الجزيرة ، في هذه الأيام نراها تلعب على أعصاب المسلمين وخاصة أنصار القاعدة وأحبائها ، وهم والحمد لله بالملايين . يا جزيرة ، الحق ينتصر فلا تنقلبي . وعيد سعيد لكل مسلم يحب الله .'),
        ('من ضمن خبركم أعلاه باستخدام القوات السورية لقصف مدنيين . . . شو هاكذب يا جماعة الخير كلها اخبار كاذبة و مقرفة و ممله أيضا الفوضى بتصحيح مسار بأي حال من الاحول فائلة و بالتالي خسرت المعارضة تأيدها للشعب السوري و ما بني على باطل فهو باطل', 'من ضمن خبركم أعلاه باستخدام القوات السورية لقصف مدنيين . . . ما هذا الكذب يا جماعة الخير ! كلها أخبار كاذبة ، ومقرفة ، ومملة . أيضا الفوضى بتصحيح مسار بأي حال من الأحوال قائمة ، وبالتالي خسرت المعارضة تأييدها للشعب السوري ، وما بني على باطل فهو باطل .'),
        ('لا اعلم تتجاهل الجزيرة الاخبار القادمة من حلب فقد قامت مظاهرتين في الجامعة احداها في الآداب وهي ضخمة اضافة الى عشرات المظاهرات القرى كما قامت الان ليلا مظاهرة حاشدة جدا بالآلاف في حي صلاح الدين وقد حصلت اشتباكات عنيفة جدا مع عناصر الامن والشبيحة', 'لا أعلم ، تتجاهل الجزيرة الأخبار القادمة من حلب ، فقد قامت مظاهرتان في الجامعة ، إحداهما في الآداب وهي ضخمة ، إضافة إلى عشرات المظاهرات في القرى . كما قامت الآن ليلا مظاهرة حاشدة جدا بالآلاف في حي صلاح الدين ، وقد حصلت اشتباكات عنيفة جدا مع عناصر الأمن والشبيحة .'),
        ('يجب على كل مسلم ان يمد يد العون لاهلنتا في سوريا و اقله يجب علينا حكومات و شعوب مقاطعة الروس و الصينيين و ازدرائهم و عدم التعامل معهم لانهم يعينون على قتل اهلنا بكل دم بارد من قبل الفأر و عصابته .', 'يجب على كل مسلم أن يمد يد العون لأهلنا في سوريا ، وأقله يجب علينا حكومات وشعوبا مقاطعة الروس والصينيين ، وازدراؤهم ، وعدم التعامل معهم ، لأنهم يعينون على قتل أهلنا بكل دم بارد من قبل الفار وعصابته .')
    ],
    'zaebuc': [
        ('اصبح العديد من الناس يستخدم وسائل التواصل الاجتماعي بشكل مفرط فمنهم من يستخدمها دون هدف و منهم من يستخدمها للتجارة و لكن لنتحدث عن اللذين يستخدمونها دون هدف لما ذلك , هذا فقط لتضييع وقت الفراغ في بعض الامور و منهم من يتعارف على ثقافات دول اخرى لكن ان تركنا ذلك قليلا سنكتشف ان هناك الكثير من الاشياء تحدث فالواقع دون ان نلاحظها , بسبب مواقع التواصل الاجتماعي لم تعد بعض العائلات تجلس مع بعضها لم يعد احد يعرف الاخر ولكن من يمكنه ان يترك هذا الادمان', 'أصبح العديد من الناس يستخدم وسائل التواصل الاجتماعي بشكل مفرط , فمنهم من يستخدمها دون هدف ومنهم من يستخدمها للتجارة , ولكن لنتحدث عن الذين يستخدمونها دون هدف . لما ذلك ? هذا فقط لتضييع وقت الفراغ في بعض الأمور ومنهم من يتعارف على ثقافات دول أخرى , لكن إن تركنا ذلك قليلا سنكتشف أن هناك الكثير من الأشياء تحدث في الواقع دون أن نلاحظها بسبب مواقع التواصل الاجتماعي . لم تعد بعض العائلات تجلس مع بعضها لم يعد أحد يعرف الآخر , ولكن من يمكنه أن يترك هذا الإدمان ?'),
        ('جميعنا , أو أغلبيتنا , نمتلك هاتفا و على هذه الهاتف هناك الكثير من مواقع التواصل الأجتماعي . هذه المواقع قد تجمعنا مع اقاربنا أو اصدقائنا أو الكثير من الناس حول العالم . و لكنها قد تلوث افكارنا و نزرع كره النفس , لكن يجب أن نتذكر أننا نحن الذين نمسك هذه الهواتف و نشعر بما نريد أن نشعر به . وسائل التواصل الإجتماعي احيانا تزيد الوعي عن الكوارث التي تحدث حول العالم و لكنها ايضا تسبب الكوارث يجب أن نحافظ على امننا ولا نعطي الجميع معلومات عن حياتنا لكي نصيح مجتمع سليم علينا الحفاظ على امانتنا و على أمانات الغير .', 'جميعنا أو أغلبيتنا نمتلك هاتفا وعلى هذا الهاتف هناك الكثير من مواقع التواصل الاجتماعي . هذه المواقع قد تجمعنا مع أقاربنا أو أصدقائنا أو الكثير من الناس حول العالم , ولكنها قد تلوث أفكارنا وتزرع كره النفس . لكن يجب أن نتذكر أننا نحن الذين نمسك هذه الهواتف ونشعر بما نريد أن نشعر به . وسائل التواصل الاجتماعي أحيانا تزيد الوعي عن الكوارث التي تحدث حول العالم ولكنها أيضا تسبب الكوارث . يجب أن نحافظ على أمننا ولا نعطي الجميع معلومات عن حياتنا . لكي نصبح مجتمعا سليما علينا الحفاظ على أمانتنا وعلى أمانات الغير .'),
        ('وسائل التواصل الاجتماعي لها تأثير كبير علي المجتمع و الفرد و من بسبب انتشارها و حب الجميع لها و إدمان الاطفال و تقليدهم للبعض لذلك يجب لمن يستخدها الانتباه و عدم نشر المقاطع او الصور التي تضر المجتمع و الاطفال . اما لبعض الافراد بدأت وسائل التواصل الاجتماعي بمساعدتهم في العديد من المجالات و منها التجارة لمنتجاتهم من قبل مشاهير وسائل التواصل الاجتماعي و دعايتهم لمنتجاتهم و نشرها حو مختلف دول العالم و الاستفاده منها , ايضا بدآ الكثيرون بالاستفاده من الشهره من خلال وسائل التواصل الاجتماعي للحصول على الشهره و المال بمختلف الطرق منها تصميم و نشر تصاميمهم في مختلف برامج التواصل الاجتماعي و ايضا بالحصول علي المال لنشر تشاجرة التجار او مختلف الشركات منها شركات المقاولات او الشركات اخرى', 'وسائل التواصل الاجتماعي لها تأثير كبير على المجتمع والفرد بسبب انتشارها وحب الجميع لها وإدمان الأطفال وتقليدهم للبعض . لذلك يجب لمن يستخدمها الانتباه وعدم نشر المقاطع أو الصور التي تضر المجتمع والأطفال . أما لبعض الأفراد بدأت وسائل التواصل الاجتماعي بمساعدتهم في العديد من المجالات , ومنها التجارة لمنتجاتهم من قبل مشاهير وسائل التواصل الاجتماعي ودعايتهم لمنتجاتهم ونشرها حول مختلف دول العالم والاستفادة منها . أيضا بدأ الكثيرون بالاستفادة من الشهرة من خلال وسائل التواصل الاجتماعي للحصول على الشهرة والمال بمختلف الطرق , منها تصميم ونشر تصاميمهم في مختلف برامج التواصل الاجتماعي . وأيضا بالحصول على المال لنشر تشاجرة التجار أو مختلف الشركات , منها شركات المقاولات أو شركات أخرى .'),
        ('وسائل التواصل الاجتماعي كثيره و سهله في التوصال مع الناس من جميع انحاح العام في عدت برامج مثل السناب جات و الفيس بوك و الانستقرام و غيره . تأثير وسائل التواصل الاجتماعي على الفرد و المجتمع عالي جدا و في كل سنه تزيد تطورات وسائل التواصل الاجتماعي فتزيد الاثر على الفرد و المجتمع . وسائل التواصل الاجتماعي سهلت على المجتمع و على الفرد جميع الاشياء الذي يجب القيام بها و المجتمع قادر على كسب معلومات و مهارات حياتيه من الاخرين خلال التواصل معهم . التواصل الاجتماعي ليس فقط التحدث مع الاصدقاء و التسليه معهم التواصل الاجتماعي له مجال في توعيت الناس من الخطر و الكثير من الاشياء المهمه في الحياة . التواصل الاجتماعي عالم ممتلئ من المعلومات الكثيره و المفيده و التواصل الاجتماتي سوف يأثر على المجتمع في السنوات القادمه بشكل كبير .', 'وسائل التواصل الاجتماعي كثيرة وسهلة في التواصل مع الناس من جميع أنحاء العالم في عدة برامج مثل السنابتشات والفيسبوك والإنستجرام وغيرها . تأثير وسائل التواصل الاجتماعي على الفرد والمجتمع عال جدا , وفي كل سنة تزيد تطورات وسائل التواصل الاجتماعي فتزيد الآثار على الفرد والمجتمع . وسائل التواصل الاجتماعي سهلت على المجتمع وعلى الفرد جميع الأشياء التي يجب القيام بها , والمجتمع قادر على كسب معلومات ومهارات حياتية من الآخرين خلال التواصل معهم . التواصل الاجتماعي ليس فقط التحدث مع الأصدقاء والتسلية معهم , التواصل الاجتماعي له مجال في توعية الناس من الخطر والكثير من الأشياء المهمة في الحياة . التواصل الاجتماعي عالم ممتلئ بالمعلومات الكثيرة والمفيدة , والتواصل الاجتماعي سوف يؤثر على المجتمع في السنوات القادمة بشكل كبير .'),
        ('هناك الكثير من وسائل التواصل الأجتماعي التي تأثر تأثيرا سلبيا وإيجابيا على الفرد وعلى المجتمع . أصبح تداول الرسائل في وسائل التواصل الأجتماعي أمرا سهلا هذة الأيام . مما آدى الى الخصام والنزاع عبر وسائل التواصل الأجتماعي بشكل يومي . أثرت وسائل التواصل الأجتماعي على المجتمع كثيرا آدى الى خصام وتفكك بعض الاسر و تشتت الأطفال . أصبح لدى كل شخص حساب على هذة الوسائل وهذا السبب ادى الى زيادة عدد الخلافات والخصام عبرهذة الوسائل . أثرت وسائل التواصل الأجتماعي تأثيرا سلبيا على المجتمع الغربي والعربي منها تعاطى المخدرات وغيرها . أثرت تأثيرا إيجابيا على فئة من المجتمع . آمل أن يكون هناك قانون أن يعاقب كل شخص يسئ بأي شكل لأي شخص بدون أي شكوى لأي مسؤل . وبذالك سوف تقل أثار وسائل التواصل الأجتماعي . في هذة الأيام كثرت هذة المشاكل في دولة الإمارات العربية المتحدة .', 'هناك الكثير من وسائل التواصل الاجتماعي التي تؤثر تأثيرا سلبيا وإيجابيا على الفرد وعلى المجتمع . أصبح تداول الرسائل في وسائل التواصل الاجتماعي أمرا سهلا هذه الأيام , مما أدى إلى الخصام والنزاع عبر وسائل التواصل الاجتماعي بشكل يومي . أثرت وسائل التواصل الاجتماعي على المجتمع كثيرا مما أدى إلى خصام وتفكك بعض الأسر وتشتت الأطفال . أصبح لدى كل شخص حساب على هذه الوسائل وهذا السبب أدى إلى زيادة عدد الخلافات والخصام عبر هذه الوسائل . أثرت وسائل التواصل الاجتماعي تأثيرا سلبيا على المجتمع الغربي والعربي منها تعاطي المخدرات وغيرها . أثرت تأثيرا إيجابيا على فئة من المجتمع . آمل أن يكون هناك قانون يعاقب كل شخص يسيء بأي شكل لأي شخص بدون أي شكوى لأي مسؤول , وبذلك سوف تقل آثار وسائل التواصل الاجتماعي . في هذه الأيام كثرت هذه المشاكل في دولة الإمارات العربية المتحدة .')
    ],
    'madar': [
        ('إزا بتريد، تنان همبرغر و تنان أهوة. بدي آخدون معي.', 'اذا بتريد، اثنين همبرغر واثنين قهوة. بدي آخذهن معي.'),
        ('لو اشتريت اتنين، حيكون تمنهم كام؟', 'لو اشتريت اثنين، حيكون ثمنهم كام؟'),
        ('في واجد لدرجة اني ماعرف اي واحد اختار.', 'فيه واجد لدرجة اني ما اعرف اي واحد اختار.'),
        ('الرحلة رقم سبع ميا و تلاتة لطوكيو نهار الجمعة، اربعة سبتمبر', 'الرحلة رقم سبعمية وثلاثة لطوكيو نهار الجمعة، اربعة سبتمبر.'),
        ('احنا مستعدين بش نجاوبو بالوفت على اي سؤال عندك.', 'احنا مستعدين باش نجاوبوا بالوقت على اي سؤال عندك.')
    ]
}



def prompt_template_gec_n_shot(n, sent, lang, examples_train):
    if lang == 'en':
        prompt_prelim = ('You are an Arabic grammatical error correction '
                        'tool that can identify and correct grammatical and spelling errors in written text.')

        prompt_instruct = ('Please identify and correct any grammatical and spelling errors in the following '
                        'sentence marked with the tag <input> SRC </input>. Make the minimal changes necessary '
                        'to correct the sentence. Do not rephrase any parts of the sentence that are already '
                        'grammatically correct, and avoid altering the meaning by adding or removing information. '
                        'After making the corrections, output the revised sentence directly without providing any '
                        'explanations. ')
        if n:
            prompt_prelim += (' We will provide you with example sentences marked with the tag <input> SRC </input>, '
                            'which contain grammatical and spelling errors. '
                            'These sentences are followed by the corrected versions, marked with <output> TGT </output>, '
                            'as reviewed and edited by human experts.')
            examples = ' Here are some in-context examples:\n'
            for i in range(n):
                src, tgt = examples_train[i]
                examples += f'({i + 1}), <input> {src} </input>: <output> {tgt} </output>.\n'
            examples += 'Please feel free to refer to these examples.\n'
            prompt_instruct += examples

        prompt_instruct += 'Remember to format the corrected output with the tag <output> Your Corrected Version </output>.'
        messages = [{'role': 'system', 'content': prompt_prelim}]
        messages.append({'role': 'user', 'content': f'{prompt_instruct} Please start: <input> {sent} </input>'})
        return messages
    
    elif lang == 'ar':
        prompt_prelim = ('أنت أداة لتصحيح الأخطاء النحوية والإملائية في اللغة العربية، حيث يمكنك تحديد'
                         ' وتصحيح الأخطاء النحوية والإملائية في النصوص المكتوبة.')

        prompt_instruct = ('يرجى تحديد وتصحيح أي أخطاء نحوية أو إملائية في الجملة التالية، المحددة بالوسم <input> النص المدخل </input>. قم بإجراء الحد الأدنى '
                            'من التعديلات اللازمة لتصحيح الجملة. لا تعد صياغة أي أجزاء صحيحة نحويا، وتجنب تغيير المعنى من خلال إضافة أو حذف أي معلومات. بعد إجراء التصحيحات، قم بإخراج '
                            'الجملة المصححة مباشرة دون أي تفسيرات.')

        if n:
            prompt_prelim += ('سنزودك بجمل تحتوي على أخطاء نحوية وإملائية، محددة بالوسم <input> النص المدخل </input>. '
                              'تلي هذه الجمل النسخ المصححة، المحددة بالوسم <output> النص المصحح </output>، والتي تمت مراجعتها وتحريرها من قبل خبراء لغويين')

            examples = ' إليك بعض الأمثلة ضمن السياق:\n'
            for i in range(n):
                src, tgt = examples_train[i]
                examples += f'({i + 1}), <input> {src} </input>: <output> {tgt} </output>.\n'
            examples += 'لا تتردد في الرجوع إلى هذه الأمثلة.\n'
            prompt_instruct += examples

        prompt_instruct += 'تذكر تنسيق النص المصحح باستخدام الوسم <output> النص المصحح </output>'

        messages = [{'role': 'system', 'content': prompt_prelim}]
        messages.append({'role': 'user', 'content': f'{prompt_instruct} الرجاء البدء: <input> {sent} </input>'})
        return messages



def prompt_template_coda_n_shot(n, sent, lang, examples_train):
    if lang == 'en':
        prompt_prelim = ('You are a dialectal Arabic text normalization '
                        'tool that can normalize dialectal Arabic text into the '
                        'Conventional Orthography for Dialectal Arabic (CODA). CODA provides a standardized system '
                        'for writing Arabic dialects, which are often written informally or phonetically. '
                        'By using CODA, your task is to convert these informal, dialectal texts into a consistent, '
                        'standardized orthographic form, making them more uniform while retaining '
                        'the nuances of the original dialect.')

        prompt_instruct = ('Please standardize the following sentence marked with the tag <input> SRC </input> into the '
                        'CODA convention. Avoid altering the meaning by adding or removing information. '
                        'Make sure the normalized output sentence is in Arabic script. '
                        'Output the normalized sentence directly without providing any explanations. ')
        if n:
            prompt_prelim += (' We will provide you with example sentences marked with the tag <input> SRC </input>, '
                            'which are written in dialectal Arabic. '
                            'These sentences are followed by their CODA versions, marked with <output> TGT </output>, '
                            'as reviewed and edited by human experts.')
            examples = ' Here are some in-context examples:\n'
            for i in range(n):
                src, tgt = examples_train[i]
                examples += f'({i + 1}), <input> {src} </input>: <output> {tgt} </output>.\n'
            examples += 'Please feel free to refer to these examples.\n'
            prompt_instruct += examples

        prompt_instruct += 'Remember to format the CODA standardized output with the tag <output> Your CODA Version </output>.'

        messages = [{'role': 'system', 'content': prompt_prelim}]
        messages.append({'role': 'user', 'content': f'{prompt_instruct} Please start: <input> {sent} </input>'})
        return messages
    
    elif lang == 'ar':
        prompt_prelim = ('أنت أداة لتصحيح النصوص العربية العامية، حيث يمكنك تصحيح النصوص العامية إلى الإملاء القياسي للعامية العربية (CODA). '
                         'يوفر CODA نظاما موحدا لكتابة اللهجات العربية التي تكتب غالبا بطريقة غير رسمية أو صوتية. باستخدام CODA، '
                         'مهمتك هي تصحيح هذه النصوص العامية غير الرسمية إلى شكل إملائي موحد ومتسق، مع الحفاظ على الفروق الدقيقة للهجة الأصلية.')

        prompt_instruct = ('يرجى تصحيح الجملة التالية المميزة بالوسم <input> النص المدخل </input> وفقا لمعيار CODA. '
                           'تجنب تغيير المعنى عن طريق إضافة أو إزالة معلومات. أخرج الجملة مباشرة دون أي تفسيرات.')

        if n:
            prompt_prelim += (' سنزودك بجمل أمثلة مميزة بالوسم <input> النص المدخل </input>، وهي مكتوبة بالعامية العربية. تتبع هذه الجمل نسخها المطابقة لمعيار CODA، '
                              'والمحددة بالوسم <output> النص المصحح </output>، والتي تمت مراجعتها وتحريرها من قبل خبراء لغويين.')

            examples = ' إليك بعض الأمثلة ضمن السياق:\n'
            for i in range(n):
                src, tgt = examples_train[i]
                examples += f'({i + 1}), <input> {src} </input>: <output> {tgt} </output>.\n'
            examples += 'لا تتردد في الرجوع إلى هذه الأمثلة.\n'
            prompt_instruct += examples

        prompt_instruct += 'تذكر تنسيق النص المصحح باستخدام الوسم <output> النص المصحح </output>'

        messages = [{'role': 'system', 'content': prompt_prelim}]
        messages.append({'role': 'user', 'content': f'{prompt_instruct} الرجاء البدء: <input> {sent} </input>'})
        return messages


def gpt_predict(arguments, model):
    sent, sent_id, dataset_name, split, output_dir, n_shot, task, lang, examples = arguments
    if task == 'gec':
        messages = prompt_template_gec_n_shot(n_shot, sent, lang, examples)
    elif task == 'coda':
        messages = prompt_template_coda_n_shot(n_shot, sent, lang, examples)
    try:
        chat = openai.chat.completions.create(model=model, messages=messages)
        pred = chat.choices[0].message.content
        print(f'Example: {sent_id} from {dataset_name}-{split} ... done!', flush=True)
    except:
        pred = sent
        print(f'ERROR! Example: {sent_id} from {dataset_name}-{split}', flush=True)

    
    output_dir = os.path.join(output_dir, dataset_name, split)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f'{sent_id}.json'), 'w') as f:
        json.dump({'input': messages, 'output': pred}, f, ensure_ascii=False)


def load_data(src_path, tgt_path):
    with open(src_path) as f:
        src_data = [line.strip() for line in f.readlines()]
    with open(tgt_path) as f:
        tgt_data = [line.strip() for line in f.readlines()]
    return {'src': src_data, 'tgt': tgt_data} 


data = {}
data_dir='/scratch/ba63/arabic-text-editing/gpt-exp-data'
qalb14_dev = load_data(src_path=f'{data_dir}/qalb14/QALB-2014-L1-Dev.sent.no_ids.clean.dediac',
                         tgt_path=f'{data_dir}/qalb14/QALB-2014-L1-Dev.cor.no_ids.dediac')
qalb14_test = load_data(src_path=f'{data_dir}/qalb14/QALB-2014-L1-Test.sent.no_ids.clean.dediac',
                         tgt_path=f'{data_dir}/qalb14/QALB-2014-L1-Test.cor.no_ids.dediac')

qalb15_test = load_data(src_path=f'{data_dir}/qalb15/QALB-2015-L1-Test.sent.no_ids.dediac',
                         tgt_path=f'{data_dir}/qalb15/QALB-2015-L1-Test.cor.no_ids.dediac')

zaebuc_dev = load_data(src_path=f'{data_dir}/zaebuc/dev.sent.raw.pnx.tok.dediac',
                         tgt_path=f'{data_dir}/zaebuc/dev.sent.cor.pnx.tok.dediac')
zaebuc_test = load_data(src_path=f'{data_dir}/zaebuc/test.sent.raw.pnx.tok.dediac',
                         tgt_path=f'{data_dir}/zaebuc/test.sent.cor.pnx.tok.dediac')

madar_dev = load_data(src_path=f'{data_dir}/madar/dev.preproc.raw.txt',
                      tgt_path=f'{data_dir}/madar/dev.preproc.coda.txt')
madar_test = load_data(src_path=f'{data_dir}/madar/test.preproc.raw.txt',
                      tgt_path=f'{data_dir}/madar/test.preproc.coda.txt')


data['qalb14'] = {'dev': qalb14_dev, 'test': qalb14_test}
data['qalb15'] = {'test': qalb15_test}
data['zaebuc'] = {'dev': zaebuc_dev, 'test': zaebuc_test}
data['madar'] = {'dev': madar_dev, 'test': madar_test}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shot", default=5,
                        type=int, help="What prompting scenario to use: zero-shot, 5-shot, etc.")
    parser.add_argument("--datasets", nargs='+', default=['qalb14', 'qalb15', 'zaebuc', 'madar'],
                        type=str, help="Path of the directory where the sheets are.")
    parser.add_argument("--split", default='dev',
                        type=str, help="Which split to use for all datasets.")
    parser.add_argument("--output_dir", default=os.path.join(OUTPUT_DIR, 'chatgpt_output_0'),
                        type=str, help="Output directory.")
    parser.add_argument("--task", default='gec', help="Task to evaluate. gec or coda")
    parser.add_argument("--model", default='gpt-3.5-turbo', help="OpenAI model.")
    parser.add_argument("--lang", default='en', help="language to prompt the model in.")
    args = parser.parse_args()
    
    GEC_DATASETS = [
        ('qalb14', args.split),
        ('zaebuc', args.split)
    ]
    
    if args.split == 'test':
        GEC_DATASETS.append(('qalb15', 'test'))

    CODA_DATASETS = [
        ('madar', args.split),
    ]

    output_dir = args.output_dir

    predictions = {}
    if args.task == 'gec':
        DATASETS = GEC_DATASETS
    elif args.task == 'coda':
        DATASETS = CODA_DATASETS

    for dataset_name, split in DATASETS:
        if dataset_name not in args.datasets:
            continue
        if dataset_name == 'qalb15':
            examples_train_ = examples_train['qalb14']
        else:
            examples_train_ = examples_train[dataset_name]
        data_ = [(sent, i, dataset_name, split, output_dir, args.n_shot, args.task, args.lang, examples_train_)
                 for i, sent in enumerate(data[dataset_name][split]['src'])]


        preds = [gpt_predict(sent_info, args.model) for sent_info in data_]

        for pred in preds:
            predictions.setdefault(dataset_name, {}).setdefault(
                split, []).append(pred)
