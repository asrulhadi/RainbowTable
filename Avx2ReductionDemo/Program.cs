using System;
using System.Threading.Tasks;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

class AvxReduction
{
    const ulong CONST1 = 0x5bd1e995UL;
    const ulong CONST2 = 0x9e3779b9UL;
    const ulong MASK40 = 0xFFFFFFFFFFUL;

    public static void Reduce(ulong[] hashes, int[] steps, int tableId, ulong[] results)
    {
        if (!Avx2.IsSupported)
            throw new PlatformNotSupportedException("AVX2 not supported on this CPU.");

        int width = Vector256<ulong>.Count; // 4
        int length = hashes.Length;

        unsafe
        {
            fixed (ulong* pHashes = hashes)
            fixed (ulong* pResults = results)
            {
                ulong* _hashes = pHashes;
                ulong* _results = pResults;
                Parallel.For(0, length / width, i =>
                {
                    int offset = i * width;

                    Vector256<ulong> hashVec = Avx.LoadVector256(_hashes + offset);
                    Vector256<ulong> stepVec = CreateStepVector(steps, offset);
                    Vector256<ulong> tableVec = Vector256.Create((ulong)tableId * CONST2);

                    Vector256<ulong> mixed = Avx2.Xor(Avx2.Xor(hashVec, stepVec), tableVec);
                    Vector256<ulong> masked = Avx2.And(mixed, Vector256.Create(MASK40));

                    Avx.Store(_results + offset, masked);
                });
            }
        }

        for (int i = (length / width) * width; i < length; i++)
        {
            ulong step = ((ulong)(steps[i] + 1) * CONST1);
            ulong mix = hashes[i] ^ step ^ ((ulong)tableId * CONST2);
            results[i] = mix & MASK40;
        }
    }

    private static Vector256<ulong> CreateStepVector(int[] steps, int offset)
    {
        return Vector256.Create(
            (ulong)(steps[offset] + 1) * CONST1,
            (ulong)(steps[offset + 1] + 1) * CONST1,
            (ulong)(steps[offset + 2] + 1) * CONST1,
            (ulong)(steps[offset + 3] + 1) * CONST1
        );
    }
}

class Program
{
    static void Main()
    {
        const int N = 1_000_000;
        ulong[] hashes = new ulong[N];
        int[] steps = new int[N];
        ulong[] results = new ulong[N];
        int tableId = 7;

        Random rng = new Random();
        for (int i = 0; i < N; i++)
        {
            hashes[i] = ((ulong)rng.Next() << 32) | (ulong)rng.Next();
            steps[i] = rng.Next(0, 1000);
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        AvxReduction.Reduce(hashes, steps, tableId, results);
        sw.Stop();
        Console.WriteLine($"Reduced {N:N0} hashes in {sw.ElapsedMilliseconds} ms");

        // Span<T> benchmark
        sw.Restart();
        BaselineReduction.ReduceSpan(hashes, steps, tableId, results);
        sw.Stop();
        Console.WriteLine($"[Span] Reduced {N:N0} hashes in {sw.ElapsedMilliseconds} ms");

        // Unsafe benchmark
        sw.Restart();
        BaselineReduction.ReduceUnsafe(hashes, steps, tableId, results);
        sw.Stop();
        Console.WriteLine($"[Unsafe] Reduced {N:N0} hashes in {sw.ElapsedMilliseconds} ms");

        Console.WriteLine($"Example result: {results[123456]:X}");
    }
}

class BaselineReduction
{
    const ulong CONST1 = 0x5bd1e995UL;
    const ulong CONST2 = 0x9e3779b9UL;
    const ulong MASK40 = 0xFFFFFFFFFFUL;

    public static void ReduceSpan(ulong[] hashes, int[] steps, int tableId, ulong[] results)
    {
        Span<ulong> h = hashes;
        Span<int> s = steps;
        Span<ulong> r = results;
        for (int i = 0; i < h.Length; i++)
        {
            ulong step = (ulong)(s[i] + 1) * CONST1;
            ulong mix = h[i] ^ step ^ ((ulong)tableId * CONST2);
            r[i] = mix & MASK40;
        }
    }

    public static void ReduceUnsafe(ulong[] hashes, int[] steps, int tableId, ulong[] results)
    {
        unsafe
        {
            fixed (ulong* h = hashes, r = results)
            fixed (int* s = steps)
            {
                for (int i = 0; i < hashes.Length; i++)
                {
                    ulong step = (ulong)(s[i] + 1) * CONST1;
                    ulong mix = h[i] ^ step ^ ((ulong)tableId * CONST2);
                    r[i] = mix & MASK40;
                }
            }
        }
    }
}