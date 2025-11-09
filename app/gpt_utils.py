
def history_summary(facts: dict) -> str:
    s = facts
    return (f"Total invoices: {s['total_invoices']}, total revenue RM {s['total_revenue']:,.0f}.  \n"
            f"Peak {s['peak_month']} (RM {s['peak_revenue']:,.0f}); "
            f"Trough {s['trough_month']} (RM {s['trough_revenue']:,.0f}).  \n"
            "Top items: " + ", ".join([t['Item No.'] for t in s['top_items']]) + ".")

def prediction_explain(facts: dict) -> str:
    i = facts['inputs']; y = facts['prediction_amount']; p = facts.get('p_high')
    base = (f"Predicted **Amount** RM {y:,.0f} for {i['Quantity']} unit(s) of "
            f"{i['Item No.']} / {i['Variant']} in {i['Month']}/{i['Year']}.  \n")
    whatif = "- What-if: +1 unit raises total roughly by the typical unit price; -1 unit lowers it similarly."
    prob = f"  High-Value probability: **{p:.1%}**." if p is not None else ""
    return base + whatif + prob
