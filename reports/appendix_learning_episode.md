## Appendix: Learning Episode Questions (TokenSmith)

Topic: **Atomic Commit and Two-Phase Commit (2PC)**

### Question 1: What problem does two-phase commit (2PC) solve in a distributed database transaction?

Correct textbook chunk(s):

“Atomicity of transactions is an important issue in building a parallel and distributed database system. If a transaction runs across two nodes, unless the system designers are careful, it may commit at one node and abort at another, leading to an inconsistent state. Transaction commit protocols ensure such a situation cannot arise. The two-phase commit protocol (2PC) is the most widely used of these protocols. The 2PC protocol is described in detail in Section 23.2.1, but the key ideas are as follows: The basic idea behind 2PC is for each node to execute the transaction until it enters the partially committed state, and then leave the commit decision to a single coordinator node; the transaction is said to be in the ready state at a node at this point. The coordinator decides to commit the transaction only if the transaction reaches the ready state at every node where it executed; otherwise (e.g., if the transaction aborts at any node), the coordinator decides to abort the transaction. Every node where the transaction executed must follow the decision of the coordinator.” (Page 990)

---

### Question 2: In two-phase commit, what are the coordinator and participant roles, and what messages are exchanged in each phase (voting + decision)?

Correct textbook chunk(s):

“When a node receives that message, it records the result (either \<commit T\> or \<abort T\>) in its log, and correspondingly either commits or aborts the transaction. Since nodes may fail, the coordinator cannot wait indefinitely for responses from all the nodes. Instead, when a prespecified interval of time has elapsed since the prepare T message was sent out, if any node has not responded to the coordinator, the coordinator can decide to abort the transaction; the steps described for aborting the transaction must be followed, just as if a node had sent an abort message for the transaction. Figure 23.2 shows an instance of successful execution of 2PC for a transaction T, with two nodes, N1 and N2, that are both willing to commit transaction T. If any of the nodes sends a no T message, the coordinator will send an abort T message to all the nodes, which will then abort the transaction.” (Pages 1102–1103)

---

### Question 3: Why is two-phase commit considered a blocking protocol? Describe the specific failure scenario that causes participants to block and what they are waiting for.

Correct textbook chunk(s):

“Software systems need to make decisions, such as the coordinator’s decision on whether to commit or abort a transaction when using 2PC… If the decision is made by a single node, such as the commit/abort decision made by a coordinator node in 2PC, the system may block in case the node fails, since other nodes have no way of determining what decision was made.” (Page 1152)

---

### Question 4: During recovery from failures, what information must be forced to stable storage for two-phase commit to guarantee correctness, and when is it forced?

Correct textbook chunk(s):

“Once the \<ready T\> log record is written, the transaction T is said to be in the ready state at the node. The ready T message is, in effect, a promise by a node to follow the coordinator’s order to commit T or to abort T. To make such a promise, the needed information must first be stored in stable storage. Otherwise, if the node crashes after sending ready T, it may be unable to make good on its promise. Further, locks acquired by the transaction must continue to be held until the transaction completes…” (Page 1103)

---

### Question 5: What is the key idea behind three-phase commit (3PC) or other non-blocking variants, and how do they avoid the blocking behavior of 2PC (at a high level)?

Correct textbook chunk(s):

“We end the section by describing how consensus can be used to make two-phase commit nonblocking.” (Page 1151)

“Extensions of the protocol that work safely under network partitioning were developed subsequently. The idea behind these extensions is similar to the majority voting idea of distributed consensus, but the protocols are specifically tailored for the task of atomic commit.” (Page 1108)

