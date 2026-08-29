(() => {
    "use strict";

    const LONG_REQUEST_SECONDS = 20;
    const MOBILE_EVIDENCE_QUERY = "(max-width: 900px)";

    const elements = {
        assistantPanel: document.getElementById("assistant-panel"),
        overviewPanel: document.getElementById("overview-panel"),
        assistantWorkspace: document.getElementById("assistant-workspace"),
        brand: document.querySelector(".brand"),
        tabs: Array.from(document.querySelectorAll('[role="tab"][data-view]')),
        panels: Array.from(document.querySelectorAll('[role="tabpanel"].view-panel')),
        newSessionButton: document.getElementById("new-session-button"),
        welcomeState: document.getElementById("welcome-state"),
        promptStarters: Array.from(document.querySelectorAll("[data-prompt]")),
        conversationScroll: document.getElementById("conversation-scroll"),
        chatLog: document.getElementById("chat-log"),
        composer: document.getElementById("composer"),
        question: document.getElementById("question"),
        submitButton: document.getElementById("submit-button"),
        requestStatus: document.getElementById("request-status"),
        requestStatusLabel: document.getElementById("request-status-label"),
        elapsedTime: document.getElementById("elapsed-time"),
        evidenceInspector: document.getElementById("evidence-inspector"),
        inspectorClose: document.getElementById("inspector-close"),
        inspectorEmpty: document.getElementById("inspector-empty"),
        inspectorContent: document.getElementById("inspector-content"),
        evidenceCount: document.getElementById("evidence-count"),
        evidenceTabs: document.getElementById("evidence-tabs"),
        evidenceDetail: document.getElementById("evidence-detail"),
        mobileEvidenceTrigger: document.getElementById("mobile-evidence-trigger"),
        mobileEvidenceLabel: document.getElementById("mobile-evidence-label"),
        sheetBackdrop: document.getElementById("sheet-backdrop"),
        globalAnnouncer: document.getElementById("global-announcer"),
        appendix: document.getElementById("analysis-appendix"),
        appendixAction: document.querySelector(".summary-action"),
        canvas: document.getElementById("evidence-canvas"),
        architectureFlow: document.querySelector(".architecture-flow[data-reveal]"),
    };

    const mobileEvidence = window.matchMedia(MOBILE_EVIDENCE_QUERY);
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");

    const state = {
        threadId: createUuid(),
        requestInFlight: false,
        evidence: [],
        selectedEvidenceIndex: 0,
        evidenceResponseId: 0,
        sheetOpen: false,
        sheetReturnFocus: null,
        requestStartedAt: 0,
        timerId: null,
        longRequestAnnounced: false,
        overviewImagesLoaded: false,
    };

    function createUuid() {
        if (typeof crypto.randomUUID === "function") {
            return crypto.randomUUID();
        }

        const bytes = new Uint8Array(16);
        crypto.getRandomValues(bytes);
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
        return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
    }

    function createElement(tagName, className, text) {
        const node = document.createElement(tagName);
        if (className) node.className = className;
        if (text !== undefined && text !== null) node.textContent = String(text);
        return node;
    }

    function announce(message) {
        elements.globalAnnouncer.textContent = "";
        window.setTimeout(() => {
            elements.globalAnnouncer.textContent = message;
        }, 30);
    }

    function pluralize(count, singular, plural = `${singular}s`) {
        return `${count} ${count === 1 ? singular : plural}`;
    }

    function formatClockTime(date = new Date()) {
        return new Intl.DateTimeFormat(undefined, {
            hour: "2-digit",
            minute: "2-digit",
        }).format(date);
    }

    function formatElapsed(seconds) {
        const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
        const remainder = Math.floor(seconds % 60).toString().padStart(2, "0");
        return `${minutes}:${remainder}`;
    }

    function activateView(panelId, focusTab = false) {
        const targetTab = elements.tabs.find((tab) => tab.dataset.view === panelId);
        const targetPanel = document.getElementById(panelId);
        if (!targetTab || !targetPanel) return;

        elements.tabs.forEach((tab) => {
            const selected = tab === targetTab;
            tab.classList.toggle("is-active", selected);
            tab.setAttribute("aria-selected", String(selected));
            tab.tabIndex = selected ? 0 : -1;
        });

        elements.panels.forEach((panel) => {
            const selected = panel === targetPanel;
            panel.hidden = !selected;
            panel.classList.toggle("is-active", selected);
            panel.classList.remove("is-entering");
        });

        window.requestAnimationFrame(() => {
            targetPanel.classList.add("is-entering");
        });
        window.setTimeout(() => targetPanel.classList.remove("is-entering"), 240);

        if (panelId === "overview-panel") {
            loadPrimaryOverviewImages();
            evidenceField.setActive(false);
        } else {
            evidenceField.setActive(!elements.welcomeState.hidden);
        }

        if (focusTab) targetTab.focus();
    }

    function handleTabKeydown(event) {
        const currentIndex = elements.tabs.indexOf(event.currentTarget);
        if (currentIndex < 0) return;

        let nextIndex = null;
        if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % elements.tabs.length;
        if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + elements.tabs.length) % elements.tabs.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = elements.tabs.length - 1;
        if (nextIndex === null) return;

        event.preventDefault();
        activateView(elements.tabs[nextIndex].dataset.view, true);
    }

    function loadImagesWithin(container) {
        if (!container) return;
        container.querySelectorAll("img[data-src]").forEach((image) => {
            image.src = image.dataset.src;
            image.removeAttribute("data-src");
        });
    }

    function loadPrimaryOverviewImages() {
        if (state.overviewImagesLoaded) return;
        loadImagesWithin(elements.overviewPanel.querySelector(".primary-figures"));
        state.overviewImagesLoaded = true;
    }

    function setWelcomeVisible(visible) {
        elements.welcomeState.hidden = !visible;
        evidenceField.setActive(visible && !elements.assistantPanel.hidden);
    }

    function resetSession() {
        if (state.requestInFlight) return;

        state.threadId = createUuid();
        state.evidence = [];
        state.selectedEvidenceIndex = 0;
        state.evidenceResponseId += 1;
        elements.chatLog.replaceChildren();
        elements.question.value = "";
        resetEvidenceInspector();
        setWelcomeVisible(true);
        evidenceField.restart();
        activateView("assistant-panel");
        elements.conversationScroll.scrollTop = 0;
        elements.question.focus();
        announce("New session started. Conversation and evidence were cleared.");
    }

    function createMessageShell(role) {
        const message = createElement("article", `message ${role}-message`);
        const header = createElement("header", "message-header");
        const roleLabel = createElement("span", "message-role");
        const marker = createElement("span", "message-marker");
        marker.setAttribute("aria-hidden", "true");
        roleLabel.append(marker, document.createTextNode(role === "user" ? "You" : "Assistant"));
        const time = createElement("time", "message-time", formatClockTime());
        time.dateTime = new Date().toISOString();
        header.append(roleLabel, time);
        message.append(header);
        return message;
    }

    function addUserMessage(question) {
        const message = createMessageShell("user");
        message.append(createElement("div", "message-body", question));
        elements.chatLog.append(message);
        scrollConversationToEnd();
    }

    function appendFormattedText(container, rawText) {
        const text = typeof rawText === "string" ? rawText.trim() : "";
        if (!text) {
            container.append(createElement("p", null, "No answer was returned."));
            return;
        }

        const blocks = text.split(/\n\s*\n/).filter(Boolean);
        blocks.forEach((block) => {
            const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
            const isList = lines.length > 0 && lines.every((line) => /^(?:[-•*]|\d+[.)])\s+/.test(line));
            if (isList) {
                const list = createElement("ul");
                lines.forEach((line) => {
                    list.append(createElement("li", null, line.replace(/^(?:[-•*]|\d+[.)])\s+/, "")));
                });
                container.append(list);
            } else {
                container.append(createElement("p", null, lines.join(" ")));
            }
        });
    }

    function createResponseCard(className, label, answer, alert = false) {
        const card = createElement("div", `response-card ${className}`);
        if (alert) card.setAttribute("role", "alert");
        card.append(createElement("p", "response-label", label));
        const content = createElement("div", "response-text");
        appendFormattedText(content, answer);
        card.append(content);
        return card;
    }

    function normalizeFdaText(value) {
        if (typeof value !== "string") return "Not included in the returned label record.";
        return value
            .replace(/\[[^\]]{1,160}\]/g, " ")
            .replace(/\s+/g, " ")
            .trim() || "Not included in the returned label record.";
    }

    function safeLabelUrl(value) {
        try {
            const url = new URL(value);
            if (url.protocol === "https:" && url.hostname === "labels.fda.gov") return url.href;
        } catch (_error) {
            return null;
        }
        return null;
    }

    function createFdaCard(fdaResult) {
        const card = createElement("div", "response-card response-fda");
        const label = createElement("p", "response-label", "FDA label reference");
        const heading = createElement("div", "fda-heading");
        heading.append(createElement("h3", null, fdaResult.drug_name || "Drug label"));

        const matchedName = fdaResult.matched_brand_name || fdaResult.matched_generic_name;
        if (matchedName) {
            heading.append(createElement("p", "fda-matched-name", `Matched label: ${matchedName}`));
        }

        const sections = createElement("div", "fda-sections");
        [
            ["Dosage and administration", fdaResult.dosage_and_administration],
            ["Contraindications", fdaResult.contraindications],
            ["Warnings", fdaResult.warnings],
        ].forEach(([title, value]) => {
            const section = createElement("section", "fda-section");
            section.append(createElement("h4", null, title));
            section.append(createElement("p", null, normalizeFdaText(value)));
            sections.append(section);
        });

        const footer = createElement("div", "fda-footer");
        footer.append(createElement("p", null, "FDA label excerpts; formatting normalized for readability. Not personalized medical advice."));
        const labelUrl = safeLabelUrl(fdaResult.label_url);
        if (labelUrl) {
            const link = createElement("a", null, "Open FDA label");
            link.href = labelUrl;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            footer.append(link);
        }

        card.append(label, heading, sections, footer);
        return card;
    }

    function createStatusPill(text, tone = "neutral") {
        return createElement("span", `status-pill status-${tone}`, text);
    }

    function appendResponseMetadata(message, data) {
        const citations = Array.isArray(data.citations) ? data.citations : [];
        const reflection = data.reflection && typeof data.reflection === "object" ? data.reflection : null;
        const hasMetadata = reflection || citations.length || data.tool_used;
        if (!hasMetadata) return;

        const metadata = createElement("div", "response-meta");
        if (reflection) {
            metadata.append(createStatusPill(
                reflection.supported ? "Source-support check passed" : "Potential source-support gap",
                reflection.supported ? "success" : "warning",
            ));
        }
        if (citations.length) {
            metadata.append(createStatusPill(pluralize(citations.length, "evidence passage"), "neutral"));
        } else if (data.tool_used) {
            metadata.append(createStatusPill(data.tool_used, "neutral"));
        }
        message.append(metadata);

        if (reflection) {
            const disclosure = createElement("details", "support-disclosure");
            disclosure.append(createElement("summary", null, "How this was checked"));
            disclosure.append(createElement(
                "p",
                null,
                "This automated check compares content words and numeric values in the generated answer with retrieved passages. It is not clinical verification and does not assess medical correctness.",
            ));
            message.append(disclosure);
        }
    }

    function addAssistantMessage(data) {
        const message = createMessageShell("assistant");
        let card;

        if (data.error) {
            card = createResponseCard("response-error", "Request error", data.error, true);
        } else if (data.fda_result) {
            card = createFdaCard(data.fda_result);
        } else if (data.needs_clarification) {
            card = createResponseCard("response-clarification", "Clarification needed", data.answer);
        } else if (data.reflection && data.reflection.supported === false) {
            card = createResponseCard("response-refusal", "Source support unresolved", data.answer);
        } else {
            card = createResponseCard("response-answer", "Evidence-grounded response", data.answer);
        }

        message.append(card);
        appendResponseMetadata(message, data);
        elements.chatLog.append(message);

        if (!data.error) renderEvidence(Array.isArray(data.citations) ? data.citations : []);
        scrollConversationToEnd();
    }

    function normalizeApiError(payload, fallback) {
        if (payload && typeof payload.error === "object" && typeof payload.error.message === "string") {
            return payload.error.message;
        }
        if (payload && typeof payload.error === "string") return payload.error;
        return fallback;
    }

    function setRequestBusy(busy) {
        state.requestInFlight = busy;
        elements.assistantWorkspace.setAttribute("aria-busy", String(busy));
        elements.submitButton.disabled = busy;
        elements.question.disabled = busy;
        elements.newSessionButton.disabled = busy;
        elements.requestStatus.hidden = !busy;

        if (busy) {
            state.requestStartedAt = Date.now();
            state.longRequestAnnounced = false;
            elements.requestStatusLabel.textContent = "Processing locally";
            elements.elapsedTime.textContent = "00:00";
            state.timerId = window.setInterval(updateElapsedTime, 1000);
        } else {
            if (state.timerId !== null) window.clearInterval(state.timerId);
            state.timerId = null;
            elements.requestStatusLabel.textContent = "Processing locally";
        }
    }

    function updateElapsedTime() {
        const seconds = Math.floor((Date.now() - state.requestStartedAt) / 1000);
        elements.elapsedTime.textContent = formatElapsed(seconds);
        if (seconds >= LONG_REQUEST_SECONDS && !state.longRequestAnnounced) {
            state.longRequestAnnounced = true;
            elements.requestStatusLabel.textContent = "Still processing locally";
            announce("Still processing locally. CPU generation can take longer.");
        }
    }

    async function requestQuestion(rawQuestion) {
        const question = typeof rawQuestion === "string" ? rawQuestion.trim() : "";
        if (!question || state.requestInFlight) return;

        setWelcomeVisible(false);
        addUserMessage(question);
        elements.question.value = "";
        setRequestBusy(true);

        try {
            const response = await fetch("/ask", {
                method: "POST",
                credentials: "same-origin",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, thread_id: state.threadId }),
            });

            let payload = null;
            try {
                payload = await response.json();
            } catch (_error) {
                payload = null;
            }

            if (!response.ok || (payload && payload.error)) {
                addAssistantMessage({
                    error: normalizeApiError(payload, "The request could not be completed. Please try again."),
                });
                return;
            }

            addAssistantMessage(payload || {
                error: "The local service returned an empty response. Please try again.",
            });
        } catch (_error) {
            addAssistantMessage({
                error: "Unable to reach the local service. Check that it is running, then try again.",
            });
        } finally {
            setRequestBusy(false);
            elements.question.disabled = false;
            elements.question.focus();
        }
    }

    function scrollConversationToEnd() {
        window.requestAnimationFrame(() => {
            elements.conversationScroll.scrollTo({
                top: elements.conversationScroll.scrollHeight,
                behavior: "auto",
            });
        });
    }

    function normalizeEvidence(citation) {
        return {
            subjectId: citation && citation.subject_id !== undefined ? String(citation.subject_id) : "Unavailable",
            admissionId: citation && citation.hadm_id !== undefined ? String(citation.hadm_id) : "Unavailable",
            chunkIndex: citation && citation.chunk_idx !== undefined ? String(citation.chunk_idx) : "Unavailable",
            score: citation && Number.isFinite(Number(citation.score)) ? Number(citation.score) : null,
            text: citation && typeof citation.chunk_text === "string" && citation.chunk_text.trim()
                ? citation.chunk_text.trim()
                : "Passage text is unavailable.",
        };
    }

    function resetEvidenceInspector() {
        state.evidence = [];
        state.selectedEvidenceIndex = 0;
        state.sheetOpen = false;
        elements.evidenceTabs.replaceChildren();
        elements.evidenceDetail.replaceChildren();
        elements.inspectorContent.hidden = true;
        elements.inspectorEmpty.hidden = false;
        elements.mobileEvidenceTrigger.hidden = true;
        elements.mobileEvidenceTrigger.setAttribute("aria-expanded", "false");
        syncEvidenceLayout();
    }

    function renderEvidence(citations) {
        state.evidence = citations.map(normalizeEvidence);
        state.selectedEvidenceIndex = 0;
        state.evidenceResponseId += 1;
        state.sheetOpen = false;

        if (!state.evidence.length) {
            resetEvidenceInspector();
            return;
        }

        elements.inspectorEmpty.hidden = true;
        elements.inspectorContent.hidden = false;
        const countText = pluralize(state.evidence.length, "evidence passage");
        elements.evidenceCount.textContent = countText;
        elements.mobileEvidenceLabel.textContent = `Review ${countText}`;
        elements.mobileEvidenceTrigger.hidden = false;
        buildEvidenceTabs();
        selectEvidence(0, false);
        syncEvidenceLayout();
    }

    function buildEvidenceTabs() {
        elements.evidenceTabs.replaceChildren();
        state.evidence.forEach((_evidence, index) => {
            const button = createElement("button", "evidence-tab", `P${index + 1}`);
            button.type = "button";
            button.id = `evidence-tab-${state.evidenceResponseId}-${index}`;
            button.setAttribute("role", "tab");
            button.setAttribute("aria-controls", "evidence-detail");
            button.setAttribute("aria-label", `Evidence passage ${index + 1}`);
            button.setAttribute("aria-selected", String(index === 0));
            button.tabIndex = index === 0 ? 0 : -1;
            button.addEventListener("click", () => selectEvidence(index, false));
            button.addEventListener("keydown", handleEvidenceTabKeydown);
            elements.evidenceTabs.append(button);
        });
    }

    function handleEvidenceTabKeydown(event) {
        const tabs = Array.from(elements.evidenceTabs.querySelectorAll('[role="tab"]'));
        const currentIndex = tabs.indexOf(event.currentTarget);
        let nextIndex = null;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (currentIndex + 1) % tabs.length;
        if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = tabs.length - 1;
        if (nextIndex === null) return;

        event.preventDefault();
        selectEvidence(nextIndex, true);
    }

    function selectEvidence(index, focusTab) {
        if (!state.evidence[index]) return;
        state.selectedEvidenceIndex = index;
        const tabs = Array.from(elements.evidenceTabs.querySelectorAll('[role="tab"]'));
        tabs.forEach((tab, tabIndex) => {
            const selected = tabIndex === index;
            tab.setAttribute("aria-selected", String(selected));
            tab.tabIndex = selected ? 0 : -1;
        });
        if (focusTab && tabs[index]) tabs[index].focus();

        const evidence = state.evidence[index];
        const header = createElement("header", "evidence-detail-header");
        const titleWrap = createElement("div");
        titleWrap.append(createElement("p", "eyebrow", `Passage ${String(index + 1).padStart(2, "0")}`));
        titleWrap.append(createElement("h3", null, "Retrieved source passage"));
        header.append(titleWrap);
        header.append(createElement(
            "span",
            "relevance-score",
            evidence.score === null ? "Relevance —" : `Relevance ${evidence.score.toFixed(3)}`,
        ));

        const identifiers = createElement("dl", "evidence-identifiers");
        [
            ["Subject ID", evidence.subjectId],
            ["Admission ID", evidence.admissionId],
            ["Chunk", evidence.chunkIndex],
            ["Measure", "Retrieval relevance"],
        ].forEach(([term, value]) => {
            const group = createElement("div");
            group.append(createElement("dt", null, term));
            group.append(createElement("dd", null, value));
            identifiers.append(group);
        });

        const passage = createElement("p", "evidence-passage", evidence.text);
        elements.evidenceDetail.setAttribute("aria-labelledby", tabs[index].id);
        elements.evidenceDetail.replaceChildren(header, identifiers, passage);
    }

    function openEvidenceSheet() {
        if (!mobileEvidence.matches || !state.evidence.length) return;
        state.sheetOpen = true;
        state.sheetReturnFocus = document.activeElement;
        syncEvidenceLayout();
        window.requestAnimationFrame(() => elements.inspectorClose.focus());
    }

    function closeEvidenceSheet(returnFocus = true) {
        if (!mobileEvidence.matches) return;
        state.sheetOpen = false;
        syncEvidenceLayout();
        if (returnFocus && state.sheetReturnFocus instanceof HTMLElement) {
            state.sheetReturnFocus.focus();
        }
        state.sheetReturnFocus = null;
    }

    function syncEvidenceLayout() {
        const isMobile = mobileEvidence.matches;
        if (isMobile) {
            const open = state.sheetOpen && state.evidence.length > 0;
            elements.evidenceInspector.hidden = !open;
            elements.sheetBackdrop.hidden = !open;
            elements.evidenceInspector.setAttribute("role", "dialog");
            elements.evidenceInspector.setAttribute("aria-modal", "true");
            elements.mobileEvidenceTrigger.setAttribute("aria-expanded", String(open));
        } else {
            state.sheetOpen = false;
            elements.evidenceInspector.hidden = false;
            elements.sheetBackdrop.hidden = true;
            elements.evidenceInspector.removeAttribute("role");
            elements.evidenceInspector.removeAttribute("aria-modal");
            elements.mobileEvidenceTrigger.setAttribute("aria-expanded", "false");
        }
    }

    function trapEvidenceFocus(event) {
        if (!state.sheetOpen || !mobileEvidence.matches || event.key !== "Tab") return;
        const focusable = Array.from(elements.evidenceInspector.querySelectorAll(
            'button:not([disabled]), a[href], [tabindex]:not([tabindex="-1"]), summary',
        )).filter((node) => !node.hidden);
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    }

    class EvidenceField {
        constructor(canvas, prefersReducedMotion, hasFinePointer) {
            this.canvas = canvas;
            this.context = canvas ? canvas.getContext("2d") : null;
            this.interactionSurface = canvas ? (canvas.closest(".constellation") || canvas) : null;
            this.prefersReducedMotion = prefersReducedMotion;
            this.hasFinePointer = hasFinePointer;
            this.active = false;
            this.intersecting = true;
            this.progress = prefersReducedMotion.matches ? 1 : 0;
            this.startedAt = 0;
            this.elapsedBeforePause = 0;
            this.frameId = null;
            this.depthFrameId = null;
            this.depthX = 0;
            this.depthY = 0;
            this.pixelRatio = 1;
            this.duration = 900;
            this.resizeObserver = null;
            this.intersectionObserver = null;

            if (!this.context) return;

            this.handleVisibility = this.handleVisibility.bind(this);
            this.handleReducedMotion = this.handleReducedMotion.bind(this);
            this.handlePointerCapability = this.handlePointerCapability.bind(this);
            this.handlePointerMove = this.handlePointerMove.bind(this);
            this.handlePointerLeave = this.handlePointerLeave.bind(this);

            document.addEventListener("visibilitychange", this.handleVisibility);
            this.prefersReducedMotion.addEventListener("change", this.handleReducedMotion);
            this.hasFinePointer.addEventListener("change", this.handlePointerCapability);
            this.interactionSurface.addEventListener("pointermove", this.handlePointerMove, { passive: true });
            this.interactionSurface.addEventListener("pointerleave", this.handlePointerLeave, { passive: true });

            if (typeof ResizeObserver === "function") {
                this.resizeObserver = new ResizeObserver(() => this.resize());
                this.resizeObserver.observe(this.canvas);
            } else {
                window.addEventListener("resize", () => this.resize());
            }

            if (typeof IntersectionObserver === "function") {
                this.intersectionObserver = new IntersectionObserver((entries) => {
                    this.intersecting = entries[0].isIntersecting;
                    this.updatePlayback();
                }, { threshold: 0.05 });
                this.intersectionObserver.observe(this.canvas);
            }

            this.updateDepthCapability();
            this.resize();
            this.syncAnimationState();
        }

        resize() {
            if (!this.context) return;
            const rect = this.canvas.getBoundingClientRect();
            if (!rect.width || !rect.height) return;
            this.pixelRatio = Math.min(window.devicePixelRatio || 1, 1.75);
            const width = Math.max(1, Math.round(rect.width * this.pixelRatio));
            const height = Math.max(1, Math.round(rect.height * this.pixelRatio));
            if (this.canvas.width !== width || this.canvas.height !== height) {
                this.canvas.width = width;
                this.canvas.height = height;
            }
            this.draw(this.progress);
        }

        setActive(active) {
            this.active = Boolean(active);
            this.updatePlayback();
        }

        restart() {
            this.cancelEntrance();
            this.cancelDepthReturn();
            this.setDepth(0, 0);
            this.progress = this.prefersReducedMotion.matches ? 1 : 0;
            this.elapsedBeforePause = 0;
            this.startedAt = 0;
            this.draw(this.progress);
            this.updatePlayback();
        }

        handleVisibility() {
            this.updatePlayback();
        }

        handleReducedMotion() {
            this.updateDepthCapability();
            if (this.prefersReducedMotion.matches) {
                this.cancelEntrance();
                this.cancelDepthReturn();
                this.setDepth(0, 0);
                this.progress = 1;
                this.elapsedBeforePause = this.duration;
                this.draw(1);
            } else if (this.progress < 1) {
                this.updatePlayback();
            }
            this.syncAnimationState();
        }

        handlePointerCapability() {
            this.updateDepthCapability();
            if (!this.depthEnabled()) {
                this.cancelDepthReturn();
                this.setDepth(0, 0);
                this.draw(this.progress);
            }
        }

        handlePointerMove(event) {
            if (!this.depthEnabled() || !this.active) return;
            const rect = this.canvas.getBoundingClientRect();
            if (!rect.width || !rect.height) return;
            this.cancelDepthReturn();
            const normalizedX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            const normalizedY = ((event.clientY - rect.top) / rect.height) * 2 - 1;
            this.setDepth(
                Math.max(-3, Math.min(3, normalizedX * 3)),
                Math.max(-3, Math.min(3, normalizedY * 3)),
            );
            this.draw(this.progress);
        }

        handlePointerLeave() {
            if (!this.depthEnabled() || (!this.depthX && !this.depthY)) return;
            const startX = this.depthX;
            const startY = this.depthY;
            const startedAt = performance.now();

            const settle = (time) => {
                const raw = Math.min(1, Math.max(0, (time - startedAt) / 180));
                const eased = 1 - Math.pow(1 - raw, 3);
                this.setDepth(startX * (1 - eased), startY * (1 - eased));
                this.draw(this.progress);
                if (raw < 1 && this.depthEnabled() && !document.hidden) {
                    this.depthFrameId = window.requestAnimationFrame(settle);
                } else {
                    this.depthFrameId = null;
                    this.setDepth(0, 0);
                    this.draw(this.progress);
                }
            };

            this.depthFrameId = window.requestAnimationFrame(settle);
        }

        depthEnabled() {
            return this.hasFinePointer.matches && !this.prefersReducedMotion.matches;
        }

        updateDepthCapability() {
            if (!this.canvas) return;
            const enabled = this.depthEnabled();
            this.canvas.dataset.depthEnabled = String(enabled);
            this.setDepth(enabled ? this.depthX : 0, enabled ? this.depthY : 0);
        }

        setDepth(x, y) {
            this.depthX = Math.max(-3, Math.min(3, Number(x) || 0));
            this.depthY = Math.max(-3, Math.min(3, Number(y) || 0));
            if (!this.canvas) return;
            this.canvas.dataset.depthX = this.depthX.toFixed(2);
            this.canvas.dataset.depthY = this.depthY.toFixed(2);
        }

        shouldPlay() {
            return this.active
                && this.intersecting
                && !document.hidden
                && !this.prefersReducedMotion.matches
                && this.progress < 1;
        }

        syncAnimationState() {
            if (!this.canvas) return;
            if (this.prefersReducedMotion.matches || this.progress >= 1) {
                this.canvas.dataset.animationState = "static";
            } else if (this.shouldPlay()) {
                this.canvas.dataset.animationState = "entering";
            } else {
                this.canvas.dataset.animationState = "paused";
            }
        }

        updatePlayback() {
            if (!this.context) return;
            if (this.shouldPlay()) {
                if (this.frameId === null) {
                    this.startedAt = performance.now() - this.elapsedBeforePause;
                    this.frameId = window.requestAnimationFrame((time) => this.tick(time));
                }
            } else if (this.frameId !== null) {
                this.elapsedBeforePause = Math.min(this.duration, performance.now() - this.startedAt);
                this.cancelEntrance();
            } else {
                this.draw(this.progress);
            }
            this.syncAnimationState();
        }

        cancelEntrance() {
            if (this.frameId !== null) window.cancelAnimationFrame(this.frameId);
            this.frameId = null;
        }

        cancelDepthReturn() {
            if (this.depthFrameId !== null) window.cancelAnimationFrame(this.depthFrameId);
            this.depthFrameId = null;
        }

        tick(time) {
            const elapsed = Math.max(0, time - this.startedAt);
            this.elapsedBeforePause = Math.min(this.duration, elapsed);
            this.progress = Math.min(1, elapsed / this.duration);
            this.draw(this.progress);
            if (this.shouldPlay()) {
                this.frameId = window.requestAnimationFrame((nextTime) => this.tick(nextTime));
            } else {
                this.frameId = null;
                this.syncAnimationState();
            }
        }

        phase(progress, start, end) {
            return Math.max(0, Math.min(1, (progress - start) / (end - start)));
        }

        shiftedPoint(x, y, depth) {
            return {
                x: x + this.depthX * this.pixelRatio * depth,
                y: y + this.depthY * this.pixelRatio * depth,
            };
        }

        draw(rawProgress) {
            if (!this.context || !this.canvas.width || !this.canvas.height) return;
            const ctx = this.context;
            const width = this.canvas.width;
            const height = this.canvas.height;
            const clamped = Math.max(0, Math.min(1, rawProgress));
            const progress = 1 - Math.pow(1 - clamped, 3);
            const lineWidth = Math.max(this.pixelRatio, width * 0.00165);
            const query = { x: width * 0.065, y: height * 0.435, w: width * 0.14, h: height * 0.13, depth: 0.14 };
            const passages = [
                { x: width * 0.285, y: height * 0.155, w: width * 0.29, h: height * 0.15, depth: 0.34, label: "EXCERPT 01" },
                { x: width * 0.255, y: height * 0.425, w: width * 0.33, h: height * 0.16, depth: 0.62, label: "EXCERPT 02" },
                { x: width * 0.29, y: height * 0.695, w: width * 0.285, h: height * 0.15, depth: 0.9, label: "EXCERPT 03" },
            ];
            const response = { x: width * 0.765, y: height * 0.39, w: width * 0.19, h: height * 0.22, depth: 0.28 };
            const reviewY = height * 0.5;
            const mergeX = width * 0.66;
            const reviewX = width * 0.71;

            ctx.clearRect(0, 0, width, height);
            ctx.lineCap = "square";
            ctx.lineJoin = "round";

            const queryProgress = this.phase(progress, 0, 0.26);
            const branchProgress = this.phase(progress, 0.08, 0.52);
            const railProgress = this.phase(progress, 0.28, 0.78);
            const responseProgress = this.phase(progress, 0.58, 0.96);

            const queryCenter = this.shiftedPoint(query.x + query.w, query.y + query.h / 2, query.depth);
            const branchX = width * 0.23;
            passages.forEach((passage, index) => {
                const passageLeft = this.shiftedPoint(passage.x, passage.y + passage.h / 2, passage.depth);
                const stagger = this.phase(branchProgress, index * 0.08, 0.76 + index * 0.08);
                this.drawPolyline(ctx, [
                    queryCenter,
                    { x: branchX, y: queryCenter.y },
                    { x: branchX, y: passageLeft.y },
                    passageLeft,
                ], stagger, "rgba(183, 200, 206, 0.16)", lineWidth);
            });

            passages.forEach((passage, index) => {
                const passageRight = this.shiftedPoint(passage.x + passage.w, passage.y + passage.h / 2, passage.depth);
                const railEnd = { x: reviewX, y: reviewY };
                const railPoints = index === 1
                    ? [passageRight, railEnd]
                    : [passageRight, { x: mergeX, y: passageRight.y }, { x: mergeX, y: reviewY }, railEnd];
                const stagger = this.phase(railProgress, index * 0.055, 0.88 + index * 0.055);
                this.drawPolyline(
                    ctx,
                    railPoints,
                    stagger,
                    index === 1 ? "rgba(98, 199, 208, 0.68)" : "rgba(183, 200, 206, 0.42)",
                    lineWidth,
                );
            });

            const responseLeft = this.shiftedPoint(response.x, response.y + response.h / 2, response.depth);
            this.drawPolyline(
                ctx,
                [{ x: reviewX + 4 * this.pixelRatio, y: reviewY }, responseLeft],
                this.phase(progress, 0.68, 0.9),
                "rgba(98, 199, 208, 0.68)",
                lineWidth,
            );

            this.drawQuery(ctx, query, queryProgress);
            passages.forEach((passage, index) => {
                this.drawPlane(ctx, passage, this.phase(progress, 0.1 + index * 0.055, 0.52 + index * 0.055), false);
            });
            this.drawReviewNode(ctx, reviewX, reviewY, this.phase(progress, 0.55, 0.86));
            this.drawPlane(ctx, response, responseProgress, true);
        }

        drawPolyline(ctx, points, progress, color, width) {
            if (progress <= 0 || points.length < 2) return;
            const lengths = [];
            let total = 0;
            for (let index = 1; index < points.length; index += 1) {
                const segment = Math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y);
                lengths.push(segment);
                total += segment;
            }
            let remaining = total * progress;
            ctx.beginPath();
            ctx.moveTo(points[0].x, points[0].y);
            for (let index = 1; index < points.length && remaining > 0; index += 1) {
                const segment = lengths[index - 1];
                if (remaining >= segment) {
                    ctx.lineTo(points[index].x, points[index].y);
                    remaining -= segment;
                } else {
                    const ratio = segment ? remaining / segment : 0;
                    ctx.lineTo(
                        points[index - 1].x + (points[index].x - points[index - 1].x) * ratio,
                        points[index - 1].y + (points[index].y - points[index - 1].y) * ratio,
                    );
                    remaining = 0;
                }
            }
            ctx.strokeStyle = color;
            ctx.lineWidth = width;
            ctx.stroke();
        }

        roundedRect(ctx, x, y, width, height, radius) {
            const r = Math.min(radius, width / 2, height / 2);
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + width - r, y);
            ctx.quadraticCurveTo(x + width, y, x + width, y + r);
            ctx.lineTo(x + width, y + height - r);
            ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
            ctx.lineTo(x + r, y + height);
            ctx.quadraticCurveTo(x, y + height, x, y + height - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();
        }

        drawQuery(ctx, plane, progress) {
            if (progress <= 0) return;
            const offset = this.shiftedPoint(0, 0, plane.depth);
            const scale = 0.97 + progress * 0.03;
            const centerX = plane.x + plane.w / 2 + offset.x;
            const centerY = plane.y + plane.h / 2 + offset.y;
            ctx.save();
            ctx.globalAlpha = progress;
            ctx.translate(centerX, centerY);
            ctx.scale(scale, scale);
            ctx.translate(-centerX, -centerY);
            this.roundedRect(ctx, plane.x + offset.x, plane.y + offset.y, plane.w, plane.h, 5 * this.pixelRatio);
            ctx.fillStyle = "rgba(11, 27, 38, 0.92)";
            ctx.fill();
            ctx.strokeStyle = "rgba(98, 199, 208, 0.56)";
            ctx.lineWidth = Math.max(this.pixelRatio, this.canvas.width * 0.00155);
            ctx.stroke();
            ctx.fillStyle = "rgba(183, 200, 206, 0.86)";
            ctx.font = `${Math.max(7, 8 * this.pixelRatio)}px \"IBM Plex Mono\", monospace`;
            ctx.fillText("QUERY", plane.x + offset.x + 9 * this.pixelRatio, plane.y + offset.y + 17 * this.pixelRatio);
            ctx.fillStyle = "rgba(98, 199, 208, 0.54)";
            ctx.fillRect(
                plane.x + offset.x + 9 * this.pixelRatio,
                plane.y + offset.y + plane.h - 13 * this.pixelRatio,
                Math.max(8 * this.pixelRatio, plane.w * 0.46),
                Math.max(1, this.pixelRatio),
            );
            ctx.restore();
        }

        drawPlane(ctx, plane, progress, response) {
            if (progress <= 0) return;
            const offset = this.shiftedPoint(0, 0, plane.depth);
            const scale = 0.965 + progress * 0.035;
            const centerX = plane.x + plane.w / 2 + offset.x;
            const centerY = plane.y + plane.h / 2 + offset.y;
            const radius = 6 * this.pixelRatio;
            const side = 4 * this.pixelRatio;

            ctx.save();
            ctx.globalAlpha = progress;
            ctx.translate(centerX, centerY);
            ctx.scale(scale, scale);
            ctx.translate(-centerX, -centerY);

            this.roundedRect(ctx, plane.x + offset.x + side, plane.y + offset.y - side, plane.w, plane.h, radius);
            ctx.fillStyle = response ? "rgba(98, 199, 208, 0.05)" : "rgba(7, 17, 25, 0.48)";
            ctx.fill();
            ctx.strokeStyle = response ? "rgba(98, 199, 208, 0.16)" : "rgba(183, 200, 206, 0.1)";
            ctx.lineWidth = Math.max(1, this.pixelRatio);
            ctx.stroke();

            this.roundedRect(ctx, plane.x + offset.x, plane.y + offset.y, plane.w, plane.h, radius);
            ctx.fillStyle = response ? "rgba(16, 39, 51, 0.94)" : "rgba(11, 27, 38, 0.96)";
            ctx.fill();
            ctx.strokeStyle = response ? "rgba(98, 199, 208, 0.62)" : "rgba(183, 200, 206, 0.34)";
            ctx.lineWidth = Math.max(this.pixelRatio, this.canvas.width * 0.00145);
            ctx.stroke();

            ctx.fillStyle = response ? "rgba(98, 199, 208, 0.9)" : "rgba(183, 200, 206, 0.73)";
            ctx.font = `${Math.max(7, 7.5 * this.pixelRatio)}px \"IBM Plex Mono\", monospace`;
            if (response) {
                ctx.fillText("REVIEWABLE", plane.x + offset.x + 10 * this.pixelRatio, plane.y + offset.y + 15 * this.pixelRatio);
                ctx.fillText("RESPONSE", plane.x + offset.x + 10 * this.pixelRatio, plane.y + offset.y + 24 * this.pixelRatio);
            } else {
                ctx.fillText(plane.label, plane.x + offset.x + 10 * this.pixelRatio, plane.y + offset.y + 16 * this.pixelRatio);
            }

            const lineX = plane.x + offset.x + 10 * this.pixelRatio;
            const firstY = plane.y + offset.y + plane.h * (response ? 0.62 : 0.52);
            const lineHeight = 7 * this.pixelRatio;
            [0.72, response ? 0.58 : 0.84, response ? 0.42 : 0.61].forEach((length, index) => {
                ctx.fillStyle = index === 0 && response
                    ? "rgba(98, 199, 208, 0.46)"
                    : "rgba(183, 200, 206, 0.24)";
                ctx.fillRect(
                    lineX,
                    firstY + index * lineHeight,
                    Math.max(8 * this.pixelRatio, (plane.w - 20 * this.pixelRatio) * length),
                    Math.max(1, this.pixelRatio),
                );
            });
            ctx.restore();
        }

        drawReviewNode(ctx, x, y, progress) {
            if (progress <= 0) return;
            const radius = 3 * this.pixelRatio;
            ctx.save();
            ctx.globalAlpha = progress;
            ctx.beginPath();
            ctx.arc(x, y, radius * (0.94 + progress * 0.06), 0, Math.PI * 2);
            ctx.fillStyle = "rgba(7, 17, 25, 0.98)";
            ctx.fill();
            ctx.strokeStyle = "rgba(98, 199, 208, 0.96)";
            ctx.lineWidth = Math.max(1.5 * this.pixelRatio, this.canvas.width * 0.0017);
            ctx.stroke();
            ctx.restore();
        }
    }

    function setupArchitectureReveal() {
        const flow = elements.architectureFlow;
        if (!flow) return;

        let observer = null;
        let revealed = false;
        const reveal = () => {
            if (revealed) return;
            revealed = true;
            flow.dataset.revealState = "revealed";
            if (observer) observer.disconnect();
        };

        if (reducedMotion.matches || typeof IntersectionObserver !== "function") {
            reveal();
        } else {
            flow.dataset.revealState = "pending";
            observer = new IntersectionObserver((entries) => {
                if (entries.some((entry) => entry.isIntersecting && entry.intersectionRatio >= 0.2)) {
                    reveal();
                }
            }, { threshold: [0.2] });
            observer.observe(flow);
        }

        reducedMotion.addEventListener("change", () => {
            if (reducedMotion.matches) reveal();
        }, { once: true });
    }

    const evidenceField = new EvidenceField(elements.canvas, reducedMotion, finePointer);

    elements.tabs.forEach((tab) => {
        tab.addEventListener("click", () => activateView(tab.dataset.view));
        tab.addEventListener("keydown", handleTabKeydown);
    });

    elements.brand.addEventListener("click", (event) => {
        event.preventDefault();
        activateView("assistant-panel", true);
    });

    elements.newSessionButton.addEventListener("click", resetSession);

    elements.promptStarters.forEach((starter) => {
        starter.addEventListener("click", () => {
            const prompt = starter.dataset.prompt || "";
            elements.question.value = prompt;
            requestQuestion(prompt);
        });
    });

    elements.composer.addEventListener("submit", (event) => {
        event.preventDefault();
        requestQuestion(elements.question.value);
    });

    elements.question.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            elements.composer.requestSubmit();
        }
    });

    elements.mobileEvidenceTrigger.addEventListener("click", openEvidenceSheet);
    elements.inspectorClose.addEventListener("click", () => closeEvidenceSheet(true));
    elements.sheetBackdrop.addEventListener("click", () => closeEvidenceSheet(true));
    elements.evidenceInspector.addEventListener("keydown", trapEvidenceFocus);

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && state.sheetOpen) {
            event.preventDefault();
            closeEvidenceSheet(true);
        }
    });

    mobileEvidence.addEventListener("change", syncEvidenceLayout);

    elements.appendix.addEventListener("toggle", () => {
        elements.appendixAction.textContent = elements.appendix.open ? "Close" : "Open";
        if (elements.appendix.open) loadImagesWithin(elements.appendix);
    });

    setupArchitectureReveal();
    resetEvidenceInspector();
    evidenceField.setActive(true);
})();
