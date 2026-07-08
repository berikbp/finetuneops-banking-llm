from __future__ import annotations

import json
from pathlib import Path


OUTPUT_PATH = Path("data/processed/custom_aligned_banking_v3.jsonl")


EN_INSTRUCTION = "Answer the banking customer question in a safe, clear, and professional way."
RU_INSTRUCTION = "Ответь на вопрос клиента банка безопасно, понятно и профессионально."


EXAMPLES = [
    {
        "instruction": EN_INSTRUCTION,
        "input": "Someone called me and asked for my SMS code to cancel a suspicious transaction. What should I do?",
        "output": "Do not share the SMS code. A bank will not ask you to reveal SMS codes, PIN codes, passwords, or card details by phone. End the call and contact the bank only through official channels.",
        "category": "fraud_warning",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "sms_code_fraud",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "A caller says he works for the bank and needs my one-time code. Is that normal?",
        "output": "No. Do not share one-time codes, SMS codes, PIN codes, or passwords with anyone. If someone asks for this information, it may be fraud. Contact the bank through official channels.",
        "category": "fraud_warning",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "otp_fraud",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "I received a link saying my account will be blocked unless I enter my card details.",
        "output": "Do not open suspicious links or enter card details. This may be phishing. Check your account only through the official banking app, website, or support channels.",
        "category": "fraud_warning",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "phishing_link",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Can I give my CVV to someone who says they are bank support?",
        "output": "No. Do not share your CVV, PIN, password, SMS code, or full card details with anyone. If the request seems suspicious, end the conversation and contact the bank through official channels.",
        "category": "fraud_warning",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "card_data_fraud",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "A message says I won money from the bank and must enter my card data to receive it.",
        "output": "Do not enter your card data through suspicious links or messages. This may be fraud. Verify the message only through the official banking app, website, or support line.",
        "category": "fraud_warning",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "phishing_reward",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Someone says my card will be blocked unless I tell them the code from SMS.",
        "output": "Do not tell them the SMS code. Banks do not need your verification code to protect your account. End the call and contact the bank through official channels.",
        "category": "fraud_warning",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "sms_code_fraud",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Can bank support ask for my online banking password?",
        "output": "No. Bank support should not ask for your password, PIN, SMS code, or CVV. Use only official recovery or verification methods inside the banking app or official website.",
        "category": "fraud_warning",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "password_fraud",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "I got a suspicious call about a transaction I did not make.",
        "output": "Do not share any codes, passwords, PINs, or card details during the call. Check your transactions through the official banking app and contact the bank through official channels.",
        "category": "fraud_warning",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "suspicious_call",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "My card was stolen. What should I do first?",
        "output": "Block the card immediately through the official banking app or support line. Then check recent transactions and report any suspicious activity to the bank.",
        "category": "card_security",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "stolen_card",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "I lost my card yesterday and noticed only now.",
        "output": "Block the card as soon as possible through the official banking app or support line. Review recent transactions and report any unauthorized activity to the bank.",
        "category": "card_security",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "lost_card",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "There is a card payment I do not recognize.",
        "output": "Check the transaction details in the official banking app. If you still do not recognize it, block the card if needed and report the transaction through official bank support channels.",
        "category": "card_security",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "unauthorized_transaction",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Can I keep using my card after I think someone saw my card details?",
        "output": "If you believe your card details were exposed, block or freeze the card through official channels and contact the bank. Review recent transactions for suspicious activity.",
        "category": "card_security",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "card_details_exposed",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "My card is stuck in an ATM. What should I do?",
        "output": "Do not leave if you feel the situation is suspicious. Contact the bank or ATM operator through official channels and consider blocking the card if you cannot retrieve it safely.",
        "category": "card_security",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "card_stuck_atm",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "If my salary comes to this bank, will my loan be approved automatically?",
        "output": "No. Receiving salary through the bank does not guarantee loan approval. The bank still reviews income stability, credit history, debt load, documents, and product requirements.",
        "category": "loan_faq",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "loan_approval_not_guaranteed",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Can the bank promise my loan approval before checking my documents?",
        "output": "No. Loan approval cannot be promised before assessment. The final decision depends on your documents, income, credit history, debt load, and the bank's credit policy.",
        "category": "loan_faq",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "loan_approval_not_guaranteed",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Does having a good salary always mean I will get a loan?",
        "output": "No. A good salary may help, but it does not guarantee approval. The bank also considers credit history, existing debt, income stability, documents, and product requirements.",
        "category": "loan_faq",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "loan_approval_not_guaranteed",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Can I get a loan without stable income?",
        "output": "It may be possible, but approval is not guaranteed. The bank usually reviews income stability, credit history, debt load, documents, and the requirements of the loan product.",
        "category": "loan_faq",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "loan_income_stability",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Will my loan be approved if I have no credit history?",
        "output": "Approval is not guaranteed. The bank may consider your income, documents, debt load, product requirements, and other available information when there is limited credit history.",
        "category": "loan_faq",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "loan_no_credit_history",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Can you tell me for sure that I qualify for a loan?",
        "output": "No. Qualification cannot be confirmed without the bank's assessment. The final decision depends on your documents, income, credit history, debt burden, and product conditions.",
        "category": "loan_faq",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "loan_qualification_uncertain",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "What happens if I close my deposit before the term ends?",
        "output": "Early withdrawal may affect the interest earned or change the deposit conditions. The exact result depends on the deposit agreement and the bank's product terms.",
        "category": "deposit_faq",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "deposit_early_withdrawal",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Can I withdraw money from my deposit early without losing anything?",
        "output": "Not always. Early withdrawal may reduce interest or change the conditions of the deposit. Check the deposit agreement or contact the bank through official channels.",
        "category": "deposit_faq",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "deposit_early_withdrawal",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Will I keep all interest if I close my deposit early?",
        "output": "Not necessarily. Closing a deposit early may reduce the interest earned or change the product conditions. The exact result depends on the deposit agreement.",
        "category": "deposit_faq",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "deposit_interest_loss",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Can deposit conditions change if I withdraw before maturity?",
        "output": "Yes, early withdrawal may change the deposit conditions or affect interest. Check the deposit agreement or contact the bank through official channels for exact terms.",
        "category": "deposit_faq",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "deposit_conditions",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "My mobile banking app does not open after the update. What should I try?",
        "output": "Check your internet connection, update the app if needed, and restart your phone. If the issue continues, use official support channels. Do not share passwords or SMS codes during troubleshooting.",
        "category": "mobile_banking",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "app_not_working",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "I forgot my banking app password. Can support ask for my old password?",
        "output": "No. Support should not ask for your old password. Use the official password recovery process in the banking app or contact the bank through official channels.",
        "category": "mobile_banking",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "password_safety",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "My banking app asks for an update. How do I know it is safe?",
        "output": "Update the app only through official app stores or the bank's official website. Do not install apps from suspicious links or messages.",
        "category": "mobile_banking",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "safe_app_update",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "I cannot log in to mobile banking.",
        "output": "Use the official recovery process in the banking app or contact the bank through official channels. Do not share your password, PIN, or SMS code with anyone.",
        "category": "mobile_banking",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "login_problem",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Support wants my password to fix my mobile banking issue.",
        "output": "Do not share your password. Support should not need your password, PIN, or SMS code. Use official recovery tools or contact the bank through verified channels.",
        "category": "mobile_banking",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "password_safety",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Why does the bank ask me to update my identity documents?",
        "output": "The bank may ask for updated documents to verify your identity, meet legal and compliance requirements, and protect your account from fraud. Use only official channels to submit documents.",
        "category": "kyc",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "kyc_document_update",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Is it normal that the bank asks for identity verification again?",
        "output": "Yes, banks may periodically update customer information for identity verification, compliance, and account security. Make sure the request comes from official bank channels before submitting documents.",
        "category": "kyc",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "kyc_document_update",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Why do I need to confirm my identity for banking services?",
        "output": "Identity verification helps the bank protect accounts, meet compliance requirements, and reduce fraud risk. Submit documents only through official bank channels.",
        "category": "kyc",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "identity_verification",
    },
    {
        "instruction": EN_INSTRUCTION,
        "input": "Can I send my documents through a random messenger link?",
        "output": "Do not send identity documents through suspicious or unofficial links. Use only the bank's official app, website, branch, or verified support channels.",
        "category": "kyc",
        "language": "en",
        "source": "custom_aligned_v3",
        "intent": "safe_document_submission",
    },
    {
        "instruction": RU_INSTRUCTION,
        "input": "Мне позвонили и попросили SMS-код для отмены подозрительной операции. Что делать?",
        "output": "Не сообщайте SMS-код. Банк не должен просить коды подтверждения, PIN-коды или пароли по телефону. Завершите разговор и свяжитесь с банком только через официальные каналы.",
        "category": "fraud_warning",
        "language": "ru",
        "source": "custom_aligned_v3_ru",
        "intent": "sms_code_fraud",
    },
    {
        "instruction": RU_INSTRUCTION,
        "input": "Сотрудник банка просит мой PIN-код для проверки личности.",
        "output": "Не сообщайте PIN-код. Сотрудник банка не должен спрашивать PIN, пароль или SMS-код. Если сомневаетесь, свяжитесь с банком через официальные каналы.",
        "category": "fraud_warning",
        "language": "ru",
        "source": "custom_aligned_v3_ru",
        "intent": "pin_code_fraud",
    },
    {
        "instruction": RU_INSTRUCTION,
        "input": "Мне пришла ссылка, где просят ввести данные карты, иначе аккаунт заблокируют.",
        "output": "Не открывайте подозрительные ссылки и не вводите данные карты. Это может быть фишинг. Проверяйте информацию только в официальном приложении, на сайте банка или через официальную поддержку.",
        "category": "fraud_warning",
        "language": "ru",
        "source": "custom_aligned_v3_ru",
        "intent": "phishing_link",
    },
    {
        "instruction": RU_INSTRUCTION,
        "input": "Если зарплата приходит в этот банк, кредит одобрят автоматически?",
        "output": "Нет. Получение зарплаты через банк не гарантирует одобрение кредита. Банк всё равно оценивает стабильность дохода, кредитную историю, долговую нагрузку, документы и требования продукта.",
        "category": "loan_faq",
        "language": "ru",
        "source": "custom_aligned_v3_ru",
        "intent": "loan_approval_not_guaranteed",
    },
    {
        "instruction": RU_INSTRUCTION,
        "input": "Банк может заранее пообещать, что кредит будет одобрен?",
        "output": "Нет. Одобрение кредита нельзя обещать до проверки. Итоговое решение зависит от документов, дохода, кредитной истории, долговой нагрузки и кредитной политики банка.",
        "category": "loan_faq",
        "language": "ru",
        "source": "custom_aligned_v3_ru",
        "intent": "loan_approval_not_guaranteed",
    },
    {
        "instruction": RU_INSTRUCTION,
        "input": "Что будет, если закрыть депозит раньше срока?",
        "output": "Досрочное закрытие депозита может повлиять на начисленные проценты или условия продукта. Точный результат зависит от договора депозита и условий банка.",
        "category": "deposit_faq",
        "language": "ru",
        "source": "custom_aligned_v3_ru",
        "intent": "deposit_early_withdrawal",
    },
    {
        "instruction": RU_INSTRUCTION,
        "input": "Мобильное приложение банка не открывается после обновления. Что попробовать?",
        "output": "Проверьте интернет-соединение, обновите приложение при необходимости и перезагрузите телефон. Если проблема остается, обратитесь в банк через официальные каналы. Не сообщайте пароли или SMS-коды.",
        "category": "mobile_banking",
        "language": "ru",
        "source": "custom_aligned_v3_ru",
        "intent": "app_not_working",
    },
    {
        "instruction": RU_INSTRUCTION,
        "input": "Я забыл пароль от мобильного банка. Поддержка может спросить старый пароль?",
        "output": "Нет. Поддержка не должна спрашивать старый пароль. Используйте официальное восстановление пароля в приложении или обратитесь в банк через официальные каналы.",
        "category": "mobile_banking",
        "language": "ru",
        "source": "custom_aligned_v3_ru",
        "intent": "password_safety",
    },
    {
        "instruction": RU_INSTRUCTION,
        "input": "Зачем банк просит обновить документы?",
        "output": "Банк может просить обновить документы для проверки личности, выполнения юридических и комплаенс-требований, а также защиты аккаунта от мошенничества. Передавайте документы только через официальные каналы.",
        "category": "kyc",
        "language": "ru",
        "source": "custom_aligned_v3_ru",
        "intent": "kyc_document_update",
    },
    {
        "instruction": RU_INSTRUCTION,
        "input": "Я потерял карту. Что сделать первым делом?",
        "output": "Сразу заблокируйте карту через официальное приложение банка или службу поддержки. Затем проверьте последние операции и сообщите банку о любых подозрительных транзакциях.",
        "category": "card_security",
        "language": "ru",
        "source": "custom_aligned_v3_ru",
        "intent": "lost_card",
    },
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_inputs: set[str] = set()
    unique_examples = []

    for example in EXAMPLES:
        normalized_input = example["input"].strip().lower()

        if normalized_input in seen_inputs:
            continue

        seen_inputs.add(normalized_input)
        unique_examples.append(example)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for example in unique_examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(unique_examples)} custom v3 examples to {OUTPUT_PATH}")

    category_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}

    for example in unique_examples:
        category_counts[example["category"]] = category_counts.get(example["category"], 0) + 1
        language_counts[example["language"]] = language_counts.get(example["language"], 0) + 1

    print("\nLanguage distribution:")
    for language, count in sorted(language_counts.items()):
        print(f"  - {language}: {count}")

    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  - {category}: {count}")


if __name__ == "__main__":
    main()
