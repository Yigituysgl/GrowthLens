import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "growthens.db")
if not os.path.exists(DB_PATH):
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    from data_generator import generate_all
    generate_all()

from crm_retention import (load_rfm_data, get_segment_summary,
                            train_churn_model, simulate_campaign_roi,
                            get_market_breakdown)
from ab_testing   import load_experiment_data, run_ztest
from pricing_opt  import (load_data, get_revenue_summary,
                           estimate_elasticity, optimize_discount,
                           get_channel_roi)

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title = "GrowthLens",
    page_icon  = "🔭",
    layout     = "wide",
)

# ── Colour palette ───────────────────────────────────────────────
GREEN  = "#1D9E75"
PURPLE = "#7F77DD"
AMBER  = "#EF9F27"
RED    = "#E05C5C"
BLUE   = "#378ADD"
SEG_COLOURS = {
    "VIP"             : BLUE,
    "Loyal"           : GREEN,
    "Needs Attention" : AMBER,
    "Lost"            : RED,
    "At Risk"         : "#E07A5F",
    "New Customer"    : PURPLE,
}

# ── Cache heavy computations ─────────────────────────────────────
@st.cache_data
def load_all():
    rfm              = load_rfm_data()
    segment_summary  = get_segment_summary(rfm)
    rfm_scored, auc, importance = train_churn_model(rfm)
    exp_raw          = load_experiment_data()
    bookings, mkt    = load_data()
    by_market, by_cat= get_revenue_summary(bookings)
    elasticity, _    = estimate_elasticity(bookings)
    disc_curve, opt  = optimize_discount(bookings, elasticity)
    ch_roi           = get_channel_roi(mkt)
    ab_results       = [run_ztest(exp_raw, e)
                        for e in exp_raw['experiment'].unique()]
    return (rfm_scored, segment_summary, auc, importance,
            exp_raw, ab_results,
            bookings, by_market, by_cat,
            elasticity, disc_curve, opt, ch_roi)

# ── Sidebar navigation ───────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/telescope.png", width=64)
st.sidebar.title("GrowthLens")
st.sidebar.caption("Growth Analytics Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Executive Summary", "CRM & Retention",
     "A/B Testing", "Pricing & Revenue"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("Built with Python · SQL · Streamlit")
st.sidebar.caption("Yigit Uysaloglu · 2025")

# ── Load data ────────────────────────────────────────────────────
with st.spinner("Loading GrowthLens data..."):
    (rfm, seg_summary, auc, importance,
     exp_raw, ab_results,
     bookings, by_market, by_cat,
     elasticity, disc_curve, opt, ch_roi) = load_all()


# ════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.title("🔭 GrowthLens — Executive Summary")
    st.caption("Travel marketplace growth analytics · 50,000 customers · 24 months")
    st.divider()

    # ── Top KPI row ──────────────────────────────────────────────
    total_customers = len(rfm)
    total_revenue   = rfm['total_spend_eur'].sum()
    vip_count       = len(rfm[rfm['rfm_segment'] == 'VIP'])
    at_risk_rev     = rfm[rfm['rfm_segment'].isin(
                          ['At Risk','Needs Attention'])]['total_spend_eur'].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers",   f"{total_customers:,}")
    k2.metric("Total Revenue",     f"€{total_revenue:,.0f}")
    k3.metric("VIP Customers",     f"{vip_count:,}",
              f"{vip_count/total_customers*100:.1f}% of base")
    k4.metric("At-Risk Revenue",   f"€{at_risk_rev:,.0f}",
              "needs retention action", delta_color="inverse")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer segments")
        seg_plot = seg_summary.copy()
        fig = px.bar(
            seg_plot, x="rfm_segment", y="customer_count",
            color="rfm_segment",
            color_discrete_map=SEG_COLOURS,
            text="customer_count",
            labels={"rfm_segment":"Segment","customer_count":"Customers"},
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, height=340,
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by segment")
        fig2 = px.pie(
            seg_summary[seg_summary['total_revenue'] > 0],
            values="total_revenue", names="rfm_segment",
            color="rfm_segment", color_discrete_map=SEG_COLOURS,
            hole=0.45,
        )
        fig2.update_layout(height=340,
                           paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Revenue by market ─────────────────────────────────────────
    st.subheader("Revenue by market")
    fig3 = px.bar(
        by_market, x="market", y="total_revenue",
        color="total_revenue", color_continuous_scale="Teal",
        text="total_revenue",
        labels={"market":"Market","total_revenue":"Revenue (€)"},
    )
    fig3.update_traces(
        texttemplate='€%{text:,.0f}', textposition='outside'
    )
    fig3.update_layout(
        showlegend=False, height=320,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — CRM & RETENTION
# ════════════════════════════════════════════════════════════════
elif page == "CRM & Retention":
    st.title("👥 CRM & Retention")
    st.caption("RFM segmentation · Churn prediction · Campaign simulator")
    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["Segment Explorer", "Churn Model", "Campaign Simulator"]
    )

    # ── Tab 1: Segment Explorer ───────────────────────────────────
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_seg = st.selectbox(
                "Select segment", rfm['rfm_segment'].unique()
            )
            seg_data = rfm[rfm['rfm_segment'] == selected_seg]
            st.metric("Customers",    f"{len(seg_data):,}")
            st.metric("Avg spend",    f"€{seg_data['total_spend_eur'].mean():,.0f}")
            st.metric("Avg bookings", f"{seg_data['total_bookings'].mean():.1f}")
            st.metric("Avg recency",  f"{seg_data['recency_days'].mean():.0f} days")

        with col2:
            st.subheader("RFM score distribution")
            fig = px.histogram(
                seg_data, x="rfm_score", nbins=10,
                color_discrete_sequence=[SEG_COLOURS.get(selected_seg, GREEN)],
                labels={"rfm_score": "RFM Score"},
            )
            fig.update_layout(height=300,
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Segment comparison")
        fig2 = px.scatter(
            rfm.sample(min(3000, len(rfm))),
            x="recency_days", y="total_spend_eur",
            color="rfm_segment", size="total_bookings",
            color_discrete_map=SEG_COLOURS,
            opacity=0.6,
            labels={"recency_days":"Days since last booking",
                    "total_spend_eur":"Total spend (€)",
                    "rfm_segment":"Segment"},
        )
        fig2.update_layout(height=380,
                           plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Churn Model ────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model AUC", f"{auc:.3f}",
                      "1.0 = perfect · 0.5 = random")
            st.subheader("Feature importance")
            fig = px.bar(
                importance, x="importance", y="feature",
                orientation='h',
                color="importance",
                color_continuous_scale="Teal",
                labels={"importance":"Importance","feature":"Feature"},
            )
            fig.update_layout(
                height=340, showlegend=False,
                coloraxis_showscale=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis={'categoryorder':'total ascending'},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Churn probability distribution")
            fig2 = px.histogram(
                rfm[rfm['churn_probability'] > 0],
                x="churn_probability", nbins=40,
                color_discrete_sequence=[PURPLE],
                labels={"churn_probability":"Churn probability"},
            )
            fig2.update_layout(height=380,
                               plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Highest churn risk customers")
        top_churn = (rfm.sort_values('churn_probability', ascending=False)
                     [['customer_id','market','rfm_segment',
                        'total_bookings','total_spend_eur','churn_probability']]
                     .head(10))
        st.dataframe(top_churn, use_container_width=True, hide_index=True)

    # ── Tab 3: Campaign Simulator ─────────────────────────────────
    with tab3:
        st.subheader("Win-back campaign simulator")
        st.caption("Adjust the sliders to model different campaign scenarios")

        c1, c2, c3 = st.columns(3)
        open_rate   = c1.slider("Email open rate %",   5,  40, 20) / 100
        rebook_rate = c2.slider("Re-booking rate %",   1,  25,  8) / 100
        avg_val     = c3.slider("Avg booking value €", 30, 300, 85)
        cost_email  = st.slider("Cost per email (€)",
                                0.01, 0.50, 0.05, step=0.01)

        roi = simulate_campaign_roi(rfm, open_rate, rebook_rate,
                                    avg_val, cost_email)

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Targeted",         f"{roi['n_targeted']:,}")
        m2.metric("Re-bookings",      f"{roi['re_bookings']:,}")
        m3.metric("Revenue recovered",f"€{roi['revenue_recovered']:,.0f}")
        m4.metric("Net gain",         f"€{roi['net_gain']:,.0f}")
        m5.metric("ROI",              f"{roi['roi_pct']:.0f}%")

        # Funnel chart
        funnel = pd.DataFrame({
            "Stage": ["Targeted","Opened","Re-booked"],
            "Count": [roi['n_targeted'],
                      roi['emails_opened'],
                      roi['re_bookings']],
        })
        fig = px.funnel(funnel, x="Count", y="Stage",
                        color_discrete_sequence=[GREEN])
        fig.update_layout(height=300,
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 3 — A/B TESTING
# ════════════════════════════════════════════════════════════════
elif page == "A/B Testing":
    st.title("🧪 A/B Testing")
    st.caption("Statistical significance · Z-test · CPA analysis")
    st.divider()

    for r in ab_results:
        sig_colour = GREEN if r['is_significant'] else RED
        sig_label  = "✓ Significant" if r['is_significant'] else "✗ Not significant"

        with st.expander(
            f"**{r['experiment'].replace('_',' ').title()}** — "
            f"Lift: {r['lift_pct']:+.1f}% — {sig_label}",
            expanded=True,
        ):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Z-score",    f"{r['z_score']}",
                        "threshold ±1.96")
            col2.metric("P-value",    f"{r['p_value']}",
                        "threshold 0.05")
            col3.metric("Lift",       f"{r['lift_pct']:+.1f}%")
            col4.metric("Significant",sig_label)

            c1, c2 = st.columns(2)
            with c1:
                # Conversion rate comparison
                comp = pd.DataFrame({
                    "Variant":         ["Control","Test"],
                    "Conversion rate": [r['rate_control_pct'],
                                        r['rate_test_pct']],
                    "CPA (€)":         [r['cpa_control'],
                                        r['cpa_test']],
                })
                fig = px.bar(
                    comp, x="Variant", y="Conversion rate",
                    color="Variant",
                    color_discrete_sequence=[BLUE, GREEN],
                    text="Conversion rate",
                    title="Conversion rate (%)",
                )
                fig.update_traces(
                    texttemplate='%{text:.3f}%', textposition='outside'
                )
                fig.update_layout(
                    showlegend=False, height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig2 = px.bar(
                    comp, x="Variant", y="CPA (€)",
                    color="Variant",
                    color_discrete_sequence=[BLUE, AMBER],
                    text="CPA (€)",
                    title="Cost per acquisition (€)",
                )
                fig2.update_traces(
                    texttemplate='€%{text:.2f}', textposition='outside'
                )
                fig2.update_layout(
                    showlegend=False, height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.info(f"**Verdict:** {r['verdict']}")
            st.success(f"**Recommendation:** {r['recommendation']}")


# ════════════════════════════════════════════════════════════════
# PAGE 4 — PRICING & REVENUE
# ════════════════════════════════════════════════════════════════
elif page == "Pricing & Revenue":
    st.title("💰 Pricing & Revenue")
    st.caption("Elasticity modelling · Discount optimisation · Channel ROI")
    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["Revenue breakdown", "Discount optimizer", "Channel ROI"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue by market")
            fig = px.bar(
                by_market, x="market", y="total_revenue",
                color="total_revenue",
                color_continuous_scale="Teal",
                text="total_revenue",
            )
            fig.update_traces(
                texttemplate='€%{text:,.0f}', textposition='outside'
            )
            fig.update_layout(
                height=360, showlegend=False,
                coloraxis_showscale=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Revenue by category")
            fig2 = px.pie(
                by_cat, values="total_revenue", names="category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.update_layout(height=360,
                               paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Discount optimisation curve")
        st.metric("Price elasticity", f"{elasticity:.3f}",
                  "1% price drop → "
                  f"{abs(elasticity):.1f}% booking uplift")

        base_price = st.slider(
            "Base price (€)", 30, 250,
            int(bookings['avg_price_eur'].mean())
        )
        base_vol   = st.slider(
            "Base monthly bookings", 500, 10000,
            int(bookings['total_bookings'].mean() * 10)
        )

        disc_curve_live, opt_live = optimize_discount(
            bookings, elasticity, base_price, base_vol
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=disc_curve_live['discount_pct'],
            y=disc_curve_live['new_revenue_eur'],
            mode='lines', name='Revenue with discount',
            line=dict(color=GREEN, width=2.5),
            fill='tozeroy', fillcolor='rgba(29,158,117,0.08)',
        ))
        fig.add_trace(go.Scatter(
            x=disc_curve_live['discount_pct'],
            y=disc_curve_live['base_revenue_eur'],
            mode='lines', name='Base revenue',
            line=dict(color=AMBER, width=1.5, dash='dash'),
        ))
        fig.add_vline(
            x=opt_live['discount_pct'],
            line_dash="dot", line_color=PURPLE,
            annotation_text=f"Optimal: {opt_live['discount_pct']:.0f}%",
        )
        fig.update_layout(
            height=380,
            xaxis_title="Discount %",
            yaxis_title="Revenue (€)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Channel performance")
        fig = px.bar(
            ch_roi, x="channel", y="total_revenue",
            color="channel",
            color_discrete_sequence=[GREEN, PURPLE, AMBER],
            text="total_revenue",
            labels={"channel":"Channel","total_revenue":"Revenue (€)"},
        )
        fig.update_traces(
            texttemplate='€%{text:,.0f}', textposition='outside'
        )
        fig.update_layout(
            showlegend=False, height=340,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ch_roi, use_container_width=True, hide_index=True)