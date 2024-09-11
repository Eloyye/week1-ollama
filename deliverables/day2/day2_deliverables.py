from asyncio import run as asyncrun

from ollama_client.ModelAnalysis import ModelAnalysis


async def run_sent_test(model_analysis: ModelAnalysis):
    sentiment_analysis_samples = [""" Just Bombed a React Interview

I finally managed to get an interview after tons of applications and immediate rejections. However, this was though a recruited who reached out to me. The job was for a pure frontend React position and I studied my buns off ahead of it. I've been working as a frontend dev with some backend chops for a few years now but only using Vue and PHP (mostly Laravel) so I spent a ton of time learning React through developing. In a couple weeks I built out a CMS from scratch using Next + Supabase and felt so confident going into the interview.

During the interview I crushed every React question thrown my way and used examples from my experience. Then the live coding part came... I had submitted a form on Codepen using React and walked through the code and made the updates they wanted. The last thing they wanted me to do was write a mock Promise and that's where I tripped up. So much of my experience in the last few years has been with some fetch API and not writing actual raw promises. I fumbled horribly and my confidence was shot so things got worse... Eventually they helped me through it and it worked but it was soul crushing.

I know there are a lot of products/platforms out there to help prepare for coding interviews but I don't know which to go with. I realize there's always going to be a "gotcha" part to these interviews so I want to prepare for the next one.

Does anybody have any recommendations or experiences with any of these platforms? Or even just stories of similar experiences :)
"""]
    await model_analysis.run_sentiment_analysis_test(sentiment_analysis_samples)

async def run_text_gen(model_analysis: ModelAnalysis):
    text_generation_samples = ["""
    Given the following sentence: "Good error handling is key to robust programs, but often dreaded by programmers because there is always one more edge case. Traditional object-oriented programming languages use special exception classes that can be thrown to break the regular control flow for immediate error reporting. Let’s take a quick look at an example that explores error-safe integer division"; Split each word in that sentence and convert it into a string like so:
    ['word1', 'word2', ...]. Only reply the result and no markdown formatting.
    """,
                               ]
    await model_analysis.run_text_generation_test(text_generation_samples)

async def run_summarization_gen(model_analysis: ModelAnalysis):
    summary_samples = ["""
    Why I Prefer Exceptions to Error Values

Good error handling is key to robust programs, but often dreaded by programmers because there is always one more edge case. Traditional object-oriented programming languages use special exception classes that can be thrown to break the regular control flow for immediate error reporting. Let’s take a quick look at an example that explores error-safe integer division:

int safeDiv(int a, int b) {
   if (b == 0)
      throw Div0(); // Exceptions are passed out-of-line
   return a / b; // Totally safe now, right?
}

Newer languages tend to favor functional-style error reporting and encode errors in their return type: For example, Go encodes the error in the return type with an (res, err) tuple, and Rust returns Result<T, E>, a sum type of result and error. Even older languages such as C++ now include error values in their standard library with std::expected:

// The error is now part of the function signature
std::expected<int, Div0> safeDiv(int a, int b) {
   if (b == 0)
      return std::unexpected(Div0());
   return a / b;
}

These functional-style errors are intended to make error handling more explicit, and to force programmers to think about errors. Still, I rarely see code that uses errors as return values fare better than exceptions. Quite the opposite, in fact. My anecdotal experience is that Rust code, for example, has more calls to unwrap than I’d like. The problem here is that simply unwrapping results will crash the program on errors that could have been a user-visible error message with exceptions.
So were result types a mistake?

It depends. They are definitely useful, but there are many cases where exceptions are a much better fit. Your programming language of choice should allow you to use exceptions. However, in these languages, exceptions somehow have an (undeserved) bad reputation and cannot be used. I argue that exceptions are easier to work with as a programmer, and result in better user-facing error messages. They’re also more performant; in an example we’ll see later, a C++ implementation using exceptions is about 4× faster than Rust.
Exceptions Are Much Easier to Work With

Keep in mind that I am writing this from the perspective of someone working on server-side code that maintains a significant amount of in-memory state. This means that I prioritize keeping the server process running, and would rather have a single request fail than the entire application crash. To meet uptime SLAs, I want to handle any failure gracefully. Yes, even the ones where the abstract machine of our programming languages fails and the real system limits show up. Memory isn’t infinite, CPUs can’t process all integers, and Santa isn’t real. While my perspective certainly isn’t fully objective, I think other systems could benefit from a similar approach. Even for front-end applications, I would rather see an error dialog than a crash and loss of state.

For server programs, we also have a good way to handle errors: Send an error message to the client, and if that doesn’t work, close the connection. Yes, even for out-of-memory situations and SIGFPE, which you get for division by zero, you can send error messages instead of panicking your whole system. While shutting everything down is certainly “safe”, it is not very productive and is a potential DoS vector.

Exceptions make this extraordinarily easy. No matter where you are, 100 call stacks deep in your little corner of the program, you detect an error, so you throw an exception. All the hard work is now done automatically in stack unwinding. On Linux, this is typically done in libgcc, which determines how to unwind through the binary with DWARF eh_frame. The beauty of unwinding is that all destructors are called, all resource handles are dropped, and locks that are currently held anywhere are released. And best of all, this magic happens behind the scenes because your compiler already emits the correct unwind info1. So, a simple try-catch block at the top of our connection handling is enough to gracefully handle the error and report it to the client.
Boilerplate

Compare this to functional-style errors, where error handling is manual and super tedious. You have to explicitly check if the return value is an error and propagate it. You write the same boilerplate if (err) return err over and over again, which just litters your code.

Let’s briefly look at some code metrics for two similarly sized database projects. CockroachDB, a Go project with explicit errors:

$ rg 'if err != nil' | wc -l
24570

Nearly 25k error handling paths! Most of these errors are artifacts of the repetitive pass-through reporting style, but there are quite a few instances that are simply used for log.Fatal(). This doesn’t look very graceful to me.

CedarDB, the project we’re working on that uses C++ exceptions:

$ rg 'catch \(' | wc -l
140

Over 100× less code! At first glance, most of it is either in tests, or in cleanup code where we want to avoid nested exceptions that would call std::terminate.

Of course, there are solutions to reduce the amount of boilerplate. In Rust, you can sugar coat the ugly syntax with the ? macro. This just hides all the tedious checking, but you still need to do all the work. Additionally, your function now returns error codes, which means you have to change the public interface and recursively change all functions that call this function. Instead of working in your little corner, your extra check now propagates to half the codebase. On the other hand, there is always the temptation of a little unwrap(). Surely that snake won’t bite you.
System errors

What if I told you that most programs already handle far fewer errors than they should? Think back to the safeDiv example, and try dividing -2³² by -1. That would be 2³², but that’s out of range for an int. Oopsie-doodles.

So, are you sure that your programs don’t crash? Do you add an error result whenever a function asserts something? But there’s much more: Allocations can fail, the stack can overflow, and arithmetic operations can overflow. Without throwing exceptions, these failure modes are often simply hidden. Calling an inconspicuous function? Stackoverflow! Oops, your server is gone. Add two values? Panic! at the Rust disco. Assigning this temporary object? Say hello to the OOM killer! Whenever any part of your code flow is user-controlled (and what isn’t?), these errors can happen.

For memory allocations, you’re mostly out of luck in Go because of its GC. I have more hope for Rust, where the Kernel integration work is slowly forcing it to grow up and build some more reliable interfaces that don’t panic, e.g. Box::try_new(). But this will require refactoring all existing code.

Still, I’m amazed that this is supposed to be the state of the art for systems programming. Heck, even Java handles this much better: Catch an OutOfMemoryError and apologize to one client instead of killing the whole server and interrupting service to thousands of clients. Exceptions can handle all these system errors gracefully. And in your own code, you don’t have to refactor the world every time you discover a new edge-case and add error checking to a corner of your code.
Exceptions Lead to Better Error Messages

But with more explicit error handling, closer to the error source, you surely get better error messages! Again, my experience is the opposite: Returned error values often have little information and lead to bad error messages. The classic example is syscalls, which usually follow C conventions. There, the return value is often just an error code -1, with the actual reason passed as a side channel via errno, for whatever reason. Getting a reasonable reason why something failed is often not possible from the error code alone, but requires significant context around the calling code. You get EINVAL. What do you do now? strerror tells you: “Invalid argument”. Great! The system probably knows exactly what went wrong, but refuses to tell you the details. So the errno alone is not really helpful. For proper error messages you need the syscall as context, but even then the errors remain obscure. Consider write(2), which could return EINVAL, either for an “object which is unsuitable for writing” or for misaligned buffers. But which one is it?

Newer languages do this a bit better, but Rust errors also fall into the error-kind trap. We parse an int somewhere, and an IntErrorKind::InvalidDigit bubbles up at the user. Here I am, parsing megabytes of CSV, and you tell me “invalid digit found in string”. Call that job satisfaction? Of course, there are non-std crates like anyhow that do this better. But in my experience, error return values only encourage bad errors. For a proper error message, you need to capture significant context in the error path, which does not fit into a simple error code. Instead, you probably need to allocate a dynamic error, just like with exceptions. For debugging purposes, you may also want to include a backtrace. Now you have exceptions with extra steps!
Exceptions Are More Performant

Exceptions have one last ace up their sleeve: They have zero overhead on success, because we separate the control flow. Instead, error return values intermingle the error and the happy path, and always require a check and branch for error handling. This is not free, and introduces overhead for each result. Let’s look at the following example, a function that recursively calculates the famous Fibonacci numbers. To avoid overflows, we report an error on large calculation depths.

The following examples use C++ code, which allows us to compare both versions like for like:

struct invalid_value { std::string reason; };

unsigned do_fib_throws(unsigned n, unsigned max_depth) {
   if (!max_depth) throw invalid_value(std::to_string(n) + " exceeds max_depth");
   if (n <= 2) return 1;
   return do_fib_throws(n - 2, max_depth - 1) + do_fib_throws(n - 1, max_depth - 1);
}

std::expected<unsigned, invalid_value> do_fib_expected(unsigned n, unsigned max_depth) {
   if (!max_depth) return std::unexpected<invalid_value>(std::to_string(n) + " exceeds max_depth");
   if (n <= 2) return 1;
   auto n2 = do_fib_expected(n - 2, max_depth - 1);
   if (!n2) return std::unexpected(n2.error());
   auto n1 = do_fib_expected(n - 1, max_depth - 1);
   if (!n1) return std::unexpected(n1.error());
   return *n1 + *n2;
}

The expected version is a bit more verbose, but essentially the same code, right? Then it should run the same!
A benchmark of fib_throws vs. fib_expected. Exception takes about 7.7ms, while expected is 37ms.
A performance evaluation of exceptions vs. error values.

With C++ exceptions, 10k iterations with n=15 run in 7.7ms. With std::expected return values, it takes 37ms, almost 5× the runtime! See: Quick Bench. A Rust version is slightly faster than the C++ expected version, but still 4× slower than the throwing version! Even when we sacrifice the error message string for performance and only return an error code in the expected, exceptions are still twice as fast. Granted, functions tend to be a bit bigger (unless you strictly follow a clean code style, but then you probably do not care about performance). But I’ve also seen significant slowdowns in real code.
Why are error return values so much slower?

The obvious reason that we have is-error checks and branches does not explain everything: You might expect your CPU to quickly learn to predict branches, which are then essentially free, right? This hides the fact that each of these checks silently consumes usually invisible resources of your CPU, such as instruction cache, µop cache, branch history buffer, reorder buffer, and so on. Since almost any function can fail (remember that allocations and math can and will fail), checking all the various errors inline takes up a significant amount of these resources that can’t be used for useful code. Exceptions are handled in a completely different code path, usually outlined in cold sections, which already gives them a solid advantage.

Another difference is that instead of returning a simple int, we now return a “fat” result object. This makes the call significantly more expensive, since we now have to shuffle the values through the stack, which requires additional setup and teardown. In the success case, we now also need to convert the “fat” result to the actual value. This can prevent compiler optimizations, such as tail-call optimization. In general, the functional-style error propagation uses more registers and stack space, which in turn leads to less inlining. Of course, you can try to work around these problems by allocating errors somewhere on the page, such as in thread-local storage. This is what exceptions do!
Throwing Exceptions Used to Be Terribly Slow

But where did exceptions get their bad reputation? I think it was mostly due to the quality of their implementation. Throwing exceptions has always been expected to be quite slow, so for a long time this code path did not get much performance work, and thus was frankly terrible. But that luckily has changed, to the benefit of the entire ecosystem. Propagation with stack unwinding is still up to an order of magnitude slower than regular returns from a function. But remember how we get useful error messages for error returns? We also capture a stack trace with the same mechanism, so better unwinding benefits both sides.

One argument for having slow exceptions is that getting a response from the user is even slower. Any error condition that bubbles up to bother a user or triggers a network message is usually fine to handle with an exception. For other checks where we have a fallback path that doesn’t require an outside decision, such as checking for a cache hit, it’s probably better to use some kind of local return code. This way, exceptions can often be avoided on critical paths, and their slower performance really doesn’t matter.

However, stack unwinding used to have a big problem: It was globally synchronized. To find the stack frame locations in the binary, we need to look at a global symbol table, which is then used to find the unwind information of the current function. As already mentioned, these symbols live in the .eh_frame section in DWARF format. This information is usually static, generated at compile time, and usually doesn’t need to be synchronized. However, programs can also load code dynamically with dlopen, e.g. to load plugins or for JIT compilation.

The problem here is that we need a mutable data structure where we can dynamically update the unwind info for the new code. To protect this structure, libgcc just slapped a mutex around this code and called it a day. Now only one single exception can be thrown concurrently, even on a 100-core machine. With high concurrency and many concurrent clients, this becomes a bottleneck. Fortunately, libgcc now has functionality to make the entire unwind process lock-free. In simple cases, such as when no dynamic stack frames are loaded, there is no reason to use the locks. The dynamic case is a bit more tricky, but our co-founder Thomas Neumann has committed an implementation of a B-tree with optimistic lock coupling2 for those dynamic code sections.

Even so, unwinding is still more expensive than it needs to be. For stack unwinding, the DWARF eh_frame tables are essentially interpreted, which leaves room for improvement. There are some academic proposals such as Reliable and Fast DWARF-Based Stack Unwinding to compile the unwind tables into native code, which promises to be an order of magnitude faster than current exceptions.
Final Thoughts

In my opinion, exceptions have several advantages over error return values:

    Exceptions provide separation of concerns by keeping the error path distinct
    Results with error codes can hide system-level errors such as out-of-memory or overflows
    Exceptions make it easy to provide root-cause context by default
    Code that uses exceptions can run faster than code with inline error returns

Coming from a language like C++, which allows all sorts of things, I’m still a bit puzzled as to why newer languages like Rust or Go don’t allow the use of exceptions. While you can definitely abuse exceptions, functional-style error values are not a one-size-fits-all solution. Whenever you’re capturing context for an error message, you want exceptions, so we should be able to use them.
    """]

    await model_analysis.run_summarization_test(summary_samples)

async def main():
    models = ['mistral:latest', 'phi3:latest', 'gemma2:2b']
    model_analysis = ModelAnalysis(models)
    await run_text_gen(model_analysis)
    # await run_sent_test(model_analysis)
    # await run_summarization_gen(model_analysis)
if __name__ == '__main__':
    asyncrun(main())